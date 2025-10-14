import asyncio
from floggit import flog
import os
import random
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import (
        McpToolset, StreamableHTTPConnectionParams)

from kg_service import _fetch_knowledge_graph, _get_knowledge_subgraph

session_service = InMemorySessionService()
APP_NAME = 'kaybee_agent'

load_dotenv()

@flog
def get_random_entity(tool_context: ToolContext):
    user_id = tool_context._invocation_context.user_id
    g = _fetch_knowledge_graph(graph_id=user_id)
    entity_id = random.choice(list(g['entities'].keys()))
    entity = g['entities'][entity_id]
    nbhd = _get_knowledge_subgraph(
            entity_ids={entity_id}, graph=g, num_hops=1)

    return {
        'entity': entity,
        'entity_neighborhood': nbhd
    }


search_agent = Agent(
    model='gemini-2.5-flash',
    name='search_agent',
    description='Retrieves information from the internet.',
    instruction="""You're a specialist in Google Search""",
    tools=[google_search],
)

PROMPT = '''
You are helping to inspect and expand a knowledge graph. Your task is simple:
    1. Get a random entity, and its neighborhood, from the knowledge graph.
    2. Search the internet for new/updated properties and relationships regarding the entity.
    3. Record the new or updated knowledge.
'''

agent = Agent(
    name="knowledge_base_agent",
    model="gemini-2.5-flash",
    instruction=PROMPT,
    tools=[
        get_random_entity,
        AgentTool(agent=search_agent),
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=os.environ['KG_MCP_SERVER']
            )
        ),
    ],
)

async def call_agent(user_id: str):
    session = await session_service.create_session(
            app_name=APP_NAME, user_id=user_id)

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    user_content = types.Content(role='user', parts=[types.Part(text='Go')])
    result = runner.run_async(
            user_id=user_id, session_id=session.id, new_message=user_content)

    # Need this line.... Is there a good replacement?
    async for event in result:
        pass


async def main(knowledge: str, tool_context: ToolContext):
    '''Curates/updates knowledge store with facts contained in the conversation.

    Args:
        knowledge (str): Any potentially new or updated knowledge encountered in the conversation.
    '''
    app_name = tool_context._invocation_context.app_name
    user_id = tool_context._invocation_context.user_id
    session_id = tool_context._invocation_context.session.id

    agent_call = call_agent(
            app_name=app_name, user_id=user_id, session_id=session_id, query=knowledge)
    asyncio.create_task(agent_call)

# If running this code as a standalone Python script, you'll need to use asyncio.run() or manage the event loop.
if __name__ == "__main__":
    asyncio.run(call_agent(user_id='107435368347260378300'))

import json # Needed for pretty printing dicts
import asyncio 

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from .subagents.knowledge_graph_agent import agent as knowledge_graph_agent

session_service = InMemorySessionService()


async def call_agent(
    app_name: str,
    user_id: str,
    session_id: str,
    query: str
):
    await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(
        agent=knowledge_graph_agent,
        app_name='kaybee_agent',
        session_service=session_service
    )

    user_content = types.Content(role='user', parts=[types.Part(text=query)])
    qwer = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_content)

    # Need this line.... Is there a good replacement?
    async for event in qwer:
        pass


async def main(knowledge: str, tool_context: ToolContext):
    '''Updates knowledge graph using facts contained in the conversation.

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
    asyncio.run(main())

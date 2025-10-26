import json
import os
from pathlib import Path

import google.auth
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search
from google.adk.tools.mcp_tool.mcp_toolset import (
        McpToolset, StreamableHTTPConnectionParams)
from google.adk.planners import BuiltInPlanner
from google.genai import types
from typing import Optional

from .subagents.flowchart_agent import agent as flowchart_agent
from .tools import expand_query

def setup_environment():
    # Load environment variables from .env file in root directory
    load_dotenv()

    # Use default project from credentials if not in .env
    try:
        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    except google.auth.exceptions.DefaultCredentialsError:
        # This will happen in the test environment.
        # The tests will set the required environment variables.
        pass

PROMPT = '''You are an AI assistant whose objective is to help sports scouts find and analyze good prospects. When you respond, make suggestions to the user, to help them in their endeavors. Whenever new information is encountered, record it in the knowledge base for future reference.'''

def process_user_input(
        callback_context: CallbackContext) -> Optional[types.Content]:
    if text := callback_context.user_content.parts[-1].text:
        graph_id = callback_context.state['graph_id']
        if kb_context := expand_query(query=text, graph_id=graph_id):
            callback_context.user_content.parts.append(kb_context)

internet_search_agent = Agent(
    model='gemini-2.5-flash',
    name='internet_search_agent',
    description='Retrieves information from the internet.',
    instruction="""You're a specialist in searching the internet.""",
    tools=[google_search],
)

root_agent = Agent(
    name="knowledge_base_agent",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=128,
        )
    ),
    instruction=PROMPT,
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=os.environ['KG_MCP_SERVER'],
                headers={
                    'x-graph-id': os.environ['DEFAULT_GRAPH_ID'],
                }
            ),
            tool_filter=[
                'curate_knowledge',
                'search_knowledge_graph'
            ],
        ),
        AgentTool(agent=internet_search_agent),
    ],
    sub_agents=[
        #flowchart_agent
    ],
    before_agent_callback=process_user_input,
)

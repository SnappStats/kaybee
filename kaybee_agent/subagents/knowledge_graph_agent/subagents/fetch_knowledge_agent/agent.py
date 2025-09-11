from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .tools import get_relevant_neighborhood as fetch_knowledge_subgraph

PROMPT = """
You are a technical research agent with access to a knowledge graph. Your goal is to retrieve what is _currently_ stored in the knowledge graph, relevant to the user's input (e.g. conversation snippets, documents, etc.).

**Workflow**
1. Examine the user input to identify all key topics and entities.
2. Use `fetch_knowledge_subgraph` tool to retrieve relevant portions of the knowledge graph. It might take a few calls to gather a complete picture, especially if the input covers multiple topics, but your final call should be comprehensive (albeit relevant to the user input).

Note that the knowledge graph might be incomplete or incorrect; that's fine. Your goal is to retrieve what is currently recorded.
"""

agent = Agent(
    name="knowledge_research_agent",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=512,
        )
    ),
    instruction=PROMPT,
    tools=[
        fetch_knowledge_subgraph
    ]
)

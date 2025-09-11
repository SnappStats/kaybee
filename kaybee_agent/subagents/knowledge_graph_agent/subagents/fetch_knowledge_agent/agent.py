from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .tools import get_relevant_neighborhood

PROMPT = """
You are a specialized agent that retrieves knowledge from a knowledge graph.

Your input is a conversation snippet, and your task is to use the `get_relevant_neighborhood` tool to retrieve any portions of the knowledge graph relevant to the topics being discussed.

Use that tool as often as needed until you're satisfied you have fetched an adequate neighborhood.
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
        get_relevant_neighborhood
    ]
)

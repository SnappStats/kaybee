from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .schemas import KnowledgeGraph
from .tools import store_graph

PROMPT = """
You are a specialized agent that updates knowledge graphs.

Given a snippet of user input and a relevant (yet potentially incomplete/incorrect) subgraph of a knowledge graph, your task is to produce a replacement for the subgraph that reflects any new or updated information from the user input.

Here's the subgraph that should be updated:

    {existing_knowledge}

The replacement subgraph must:
-   **Include all new/updated knowledge** suggested by the user input.
-   **Preserve existing knowledge**, including relationships with _external_ entities (connected to, but not included in, the given subgraph, to the extent it is not updated by new knowledge.
-   **(Faithfully) simplify graph topology**, such as combining nodes that represent the same entity, or removing redundant relationships.

If there is no new or updated knowledge, the replacement subgraph should resemble the original subgraph. If the updated knowledge eradicates the existing knowledge, the replacement subgraph should be empty.

You must output the updated subgraph as a `KnowledgeGraph` object.
"""

agent = Agent(
    name="merge_knowledge_agent",
    model="gemini-2.5-flash",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=512,
        )
    ),
    instruction=PROMPT,
    output_schema=KnowledgeGraph,
    output_key='updated_knowledge',
    after_model_callback=store_graph
)

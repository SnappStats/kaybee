from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .subagents.new_knowledge_agent import agent as new_knowledge_agent
from .subagents.existing_knowledge_agent import agent as existing_knowledge_agent
from .subagents.merge_knowledge_agent import agent as merge_knowledge_agent

root_agent = Agent(
    name="knowledge_graph_agent",
    model="gemini-1.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        )
    ),
    instruction="""You are an expert at maintaining a knowledge graph.

Your task is to process a user's request and update the knowledge graph accordingly.

To do this, you have access to the following sub-agents:

- `new_knowledge_agent`: This agent extracts new information from the user's request.
- `existing_knowledge_agent`: This agent retrieves existing knowledge from the knowledge graph that is relevant to the user's request.
- `merge_knowledge_agent`: This agent merges the new information with the existing knowledge and updates the knowledge graph.

Here is the general workflow:

1.  Use the `new_knowledge_agent` to extract new information from the user's request.
2.  Use the `existing_knowledge_agent` to retrieve relevant existing knowledge.
3.  Use the `merge_knowledge_agent` to merge the new and existing knowledge.

However, you are not strictly bound to this workflow. You can use the sub-agents in any order you deem necessary to best fulfill the user's request. For example, you may need to call the `existing_knowledge_agent` multiple times to gather sufficient context before merging.
""",
    sub_agents=[
        new_knowledge_agent,
        existing_knowledge_agent,
        merge_knowledge_agent,
    ],
)

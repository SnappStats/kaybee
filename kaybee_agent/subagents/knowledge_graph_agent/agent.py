from google.adk.agents import SequentialAgent

from .subagents.fetch_knowledge_agent import agent as fetch_knowledge_agent
from .subagents.update_knowledge_agent import agent as update_knowledge_agent

agent = SequentialAgent(
    name="knowledge_graph_agent",
    description='This agent maintains a knowledge graph about topics the user cares about, and should be summoned whenever potentially new knowledge is encountered, or existing knowledge needs to be updated.',
    sub_agents=[
        fetch_knowledge_agent,
        update_knowledge_agent
    ],
)

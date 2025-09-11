from google.adk.agents import SequentialAgent

from .subagents.fetch_knowledge_agent import agent as fetch_knowledge_agent
from .subagents.update_knowledge_agent import agent as update_knowledge_agent

agent = SequentialAgent(
    name="knowledge_graph_agent",
    description='This agent maintains the knowledge graph, and should be summoned whenever potentially new knowledge is encountered.',
    sub_agents=[
        fetch_knowledge_agent,
        update_knowledge_agent
    ],
)

from google.adk.agents import SequentialAgent

from .subagents.fetch_knowledge_agent import agent as fetch_knowledge_agent
from .subagents.update_knowledge_agent import agent as update_knowledge_agent

agent = SequentialAgent(
    name="knowledge_graph_agent",
    description='This agent examines user input in order to curate a knowledge graph of facts and relationship relevant to the user and their world. The knowledge contained therein should be "special" facts not otherwise known to the world. For example, the user\'s family, desires, courses, belong in the knowledge graph, whereas a list of the US presidents does not. If the user is, say, an SME at a technology company, the knowledge graph should contain tribal knowledge for the company, such as model numbers, sub models and their relationships, project owners, marketing campaigns, and the like. Do not summon this agent to curate knowledge that is already contained in the knowledge graph.',
    sub_agents=[
        fetch_knowledge_agent,
        update_knowledge_agent
    ],
)

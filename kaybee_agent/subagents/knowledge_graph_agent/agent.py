from google.adk.agents import SequentialAgent

from .subagents.fetch_knowledge_agent import agent as fetch_knowledge_agent
from .subagents.update_knowledge_agent import agent as update_knowledge_agent

agent = SequentialAgent(
    name="knowledge_graph_agent",
    description='This agent maintains a knowledge graph of facts about the user and their world, for the purpose of carrying a conversation and answering questions. The knowledge contained therein should be "special" facts not otherwise known to the world. For example, the user\'s family, desires, courses, belong in the knowledge graph, whereas a list of the presidents does not. If the user is, say, an SME as a technology company, the knowledge graph should contain tribal knowledge for the company, such as model numbers, sub models and their relationships, project owners, marketing campaigns, and the like.',
    sub_agents=[
        fetch_knowledge_agent,
        update_knowledge_agent
    ],
)

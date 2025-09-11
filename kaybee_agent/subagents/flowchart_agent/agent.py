from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types
from .prompt import PROMPT

agent = Agent(
    name="flowchart_agent",
    model="gemini-2.5-flash",
    description='This agent is an expert at creating flowchart whenever it encounters a technical procedure.',
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        )
    ),
    instruction=PROMPT,
)

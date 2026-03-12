from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chef_researcher = Agent(
    role='Recipe Researcher',
    goal='Research and find the best recipes and cooking techniques',
    backstory="""You are an expert culinary researcher with deep knowledge of 
    international cuisines, cooking techniques, and recipe development. You excel 
    at finding creative recipe ideas and understanding flavor combinations.""",
    verbose=True,
    allow_delegation=False
)

chef_writer = Agent(
    role='Recipe Writer',
    goal='Write clear, detailed, and engaging recipes',
    backstory="""You are a professional recipe writer who creates easy-to-follow 
    recipes with precise measurements and clear instructions. You make cooking 
    accessible and enjoyable for home cooks.""",
    verbose=True,
    allow_delegation=False
)

research_task = Task(
    description="""Research a delicious pasta carbonara recipe. 
    Find the traditional Italian method and key ingredients.""",
    agent=chef_researcher
)

writing_task = Task(
    description="""Using the research, write a complete recipe for pasta carbonara.
    Include ingredients list with measurements, step-by-step instructions, 
    cooking time, and serving suggestions.""",
    agent=chef_writer
)

crew = Crew(
    agents=[chef_researcher, chef_writer],
    tasks=[research_task, writing_task],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()

print("\n\n########################")
print("## Recipe Result:")
print("########################\n")
print(result)

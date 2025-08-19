import json
import time
from pydantic import BaseModel
from typing import List, Dict, Literal
from crewai import Agent, LLM, Crew, Task, Process
from langchain_ollama import OllamaLLM

from stars_agent import StarsAgent
from utils import timeout, time_monitor


STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."


class StarsAgentTrack2(StarsAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @timeout(seconds=150)
    def call_with_multi_agents(self, observation: str) -> str:
        ollama_llm = LLM(
            model=f"ollama/{self.model_name}",
            base_url="http://localhost:11434"
        )
        current_player = Agent(
            role="Main Game Player",
            goal="Generate action according to game instructions and current game observation",
            backstory="Experienced game player, you understand game well and can generate analysis about the game",
            llm=ollama_llm,
            allow_delegation=True,
            verbose=True
        )
        teammate_player = Agent(
            role="Teammate Player",
            goal="Generate action according to game instructions and current game observation, you know your teammates well",
            backstory="Experienced game player, you understand game well and can generate analysis about the game",
            llm=ollama_llm,
            allow_delegation=True,
            verbose=True
        )
        opponent_player = Agent(
            role="Opponent Imitation Player",
            goal="Analysis opponent's moves and analyze his strategy and simulate",
            backstory="Experienced game player, you understand game well and can generate analysis about the game",
            llm=ollama_llm,
            allow_delegation=True,
            verbose=True
        )
        game_manager = Agent(
            role="Game Manager",
            goal="Generate the best action for each round in the game",
            backstory="You are the manager of your team, your team will play games together. Your crew contains a Main game player, who will make positive moves."
                      "A Teammate player, who will play and think as the teammate if there's teammates in the game. An opponent imitation player, "
                      "who will analyze opponent's moves and guess his strategy and simulate.  Make the best of them for each round in different games, and win it!"
                      "Only focus on your current action and next action, if next action is from our teammate, involve teammate player to work together. "
                      "If next action is from opponent, involve opponent imitation player to work together.  "
                      "You know how to generate a group of possible answers and simulate them with opponents or teammates according to different cases. And choose the best of them as final action!!",
            llm=ollama_llm,
            allow_delegation=True,
            verbose=True
        )

        game_task = Task(
            description=f"""This is current game observation (also may contains history) \n{observation}\n. Make an action / message that meets game requirements!! and has a high possibility to win!
""",
            expected_output="""An action / message that meets game requirements!! and has a high possibility to win!
                    """
        )

        crew = Crew(
            agents=[current_player, teammate_player, opponent_player],
            tasks=[game_task],
            manager_agent=game_manager,
            process=Process.hierarchical,
            verbose=True
        )
        generation = crew.kickoff()
        return  generation.raw

    def call_with_single_agent(self, observation: str) -> str:
        ollama_llm = LLM(
            model=f"ollama/{self.model_name}",
            base_url="http://localhost:11434"
        )
        default_player = Agent(
            role="Game Player",
            goal="Generate action according to game instructions and current game observation",
            backstory="Experienced game player, you understand game role well and can generate a reasonable answer",
            llm=ollama_llm,
            allow_delegation=False,
            verbose=True
        )

        generation = default_player.kickoff(messages=observation)
        return generation.raw

    @time_monitor(log_file="stars_agent_track2.txt")
    def __call__(self, observation: str) -> str:
        try:
            generation = self.call_with_multi_agents(observation)
        except Exception as e:
            print(f"error {e}")
            generation = self.call_with_single_agent(observation)

        thinking, content = self.generate(prompt=f"""According to current game observation: \n "{observation}"\n, understand its requirements and output format. And here is your team's solution output "{generation}". You only need to make sure it meets game requirement. 
For example, if format contains special format, make sure our output contains it. If our output contains analysis or other symbols, remove them!! For example, if your team's output is $$[force 4]$$, you may return [force 4], because "$$" is not needed. 
""", system="You are the examiner of your team. your team has already decide a final solution action for the game, you just need to examine it, make sure it looks like the sample format in the observation, and return this action with no analysis or greeting."
            "For example,  [A4 B2 C2], [wind 2], [1 defect], ...", )
        return content


if __name__ == "__main__":

    agent = StarsAgentTrack2("qwen3:8b")
    answer = agent("""
Current observations: 
[GAME] You are Player 0 in a 3-player Iterated Prisoner's Dilemma. The match lasts 5 rounds.
Round structure:
• 1 free-chat turns
• 1 decision turn - submit one token per opponent: '[<opp-id> cooperate]' or '[<opp-id> defect]' (i.e. '[1 defect] [2 cooperate]'; the default is 'cooperate'). 
Pair-wise payoff matrix (applied to each unordered pair):
  - Both cooperate  ->  3
  - Both defect     ->  1
  - You defect, they cooperate -> 5
  - You cooperate, they defect -> 0
The player(s) with the highest score at the end of all rounds wins.

[GAME] ─── Starting Round 1 ───	You can converse freely for the next 1 rounds.
Please enter the action: 
""")
    print(answer)
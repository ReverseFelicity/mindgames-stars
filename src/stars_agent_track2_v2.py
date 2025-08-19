import json
import time
from pydantic import BaseModel
from typing import List, Dict, Literal
from crewai import Agent, LLM, Crew, Task, Process
from langchain_ollama import OllamaLLM

from stars_agent import StarsAgent
from utils import timeout, time_monitor


STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."


class ColonelBlottoAction(BaseModel):
    A: int
    B: int
    C: int

class Thinking(BaseModel):
    step_1: str
    step_2: str = ""
    step_3: str = ""


class GameObservation(BaseModel):
    description_and_rule: str
    winning_condition: str
    current_player_role_or_name: str
    next_player: Literal["opponent", "teammate"]
    next_player_role_or_name: str


class StarsAgentTrack2V2(StarsAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def generate_with_format(self, prompt: str, output_format: dict, system: str=None, options: dict=None):
        thinking, content = self.generate(prompt=prompt, system=system)
        _, content = self.generate(prompt=f"rewrite this content: '{content}' into target format", system=system, output_format=output_format)
        return content

    @time_monitor(log_file="stars_agent_track2.txt")
    def __call__(self, observation: str) -> str:

        content = self.generate_with_format(prompt=f"""According to current game observation: \n "{observation}"\n, understand it and extract information from it, e.g. game description and rule,
winning condition, the role or name of current player (which is you) and the role or name of next player and if he is opponent or teammate    
""", system=STANDARD_GAME_PROMPT, output_format=GameObservation.model_json_schema())
        return content


if __name__ == "__main__":

    agent = StarsAgentTrack2V2("qwen3:8b")

    for observation in [
"""
Current observations: 
[GAME] You are Commander Alpha in a game of ColonelBlotto. Each round, you have to allocate exactly 20 units across fields: A, B, C
Format: '[A4 B2 C2]'
Win the majority of fields to win the round!
[GAME] === COLONEL BLOTTO - Round 1/9 ===
Rounds Won - Commander Alpha: 0, Commander Beta: 0
Available fields: A, B, C
Units to allocate: 20
Format: '[A4 B2 C2]'.
Please enter the action: 
""",

"""
Current observations: 
[GAME] You are playing Codenames, a 2v2 word deduction game. Each team (Red and Blue) has a Spymaster and an Operative.
Rules:
1. The Spymaster gives a one-word clue + number (e.g., '[wind 2]') based on the team's secret words (the clue may not contain any of the words on the board).
2. The Operative guesses up to N+1 words (e.g., '[breeze]') based on the clue. They can also '[pass]'.
3. Avoid guessing opponent words, neutral words (N), or the Assassin (A), which causes instant loss.
4. First team to guess all their words wins.

You are Player 0, the Spymaster for Red team. Give a one-word clue and number.
[GAME] Codenames Words:
voice    R 
field    R 
friend   N 
view     A 
smash    R 
bit      R 
square   N 
list     B 
step     B 
hour     R 
cut      B 
harmony  R 
fowl     B 
ball     B 
hate     R 
bed      N 
bread    N 
rat      B 
trick    R 
brake    N 
pocket   B 
animal   N 
roof     R 
tongue   N 
cotton   B 

Please enter the action: 
""",

"""
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
"""
    ]:
        content = agent.generate_with_format(prompt=f"""According to current game observation: \n "{observation}"\n, understand it and extract information from it, e.g. game description and rule,
winning condition, the role or name of current player (which is you) and the role or name of next player and if he is opponent or teammate    
""", system=STANDARD_GAME_PROMPT, output_format=GameObservation.model_json_schema())
        print(content)
        game_observation = GameObservation(**json.loads(content))
        print(game_observation)

        content = agent.generate_with_format(prompt=f"""Now according to the game observation:\n "{observation}"\n. You have understand the rule {game_observation.description_and_rule}.
and winning condition {game_observation.winning_condition}. And you know next player is your teammate or opponent: {game_observation.next_player}. So try your best to think about a 2-3 steps process. Following these steps
you can get a best action, but right now, only think about details of these steps
        """, system=STANDARD_GAME_PROMPT, output_format=Thinking.model_json_schema())
        print(content)
        thinking = Thinking(**json.loads(content))
        print(thinking)
import json
import time
from pydantic import BaseModel
from typing import List, Dict, Literal
from crewai import Agent, LLM, Crew, Task, Process
from langchain_ollama import OllamaLLM

from stars_agent import StarsAgent
from utils import timeout, time_monitor


ACTION_FORMAT = {
    "ColonelBlotto": {
        "Commander": {
            "command": {
                "description": "allocate exactly 20 units across fields:A, B, C\nFormat: '[A8 B8 C4]'\nWin the majority of fields to win the round!",
                "sample_action": "[A8 B8 C4]"
            }
        }
    },
    "Iterated Prisoner's Dilemma": {
        "Player": {
            "converse": {
                "description": "In free-chat turns, chat with text to express your idea or suggestion",
                "sample_action": "hi let's cooperate"
            },
            "command": {
                "description": "In decision turn - submit one token per opponent (FOR ALL OPPONENT!): '[<opp-id> cooperate]' or '[<opp-id> defect]' (i.e. '[1 defect] [2 cooperate]'; the default is 'cooperate'). Pair-wise payoff matrix (applied to each unordered pair)",
                "sample_action": "[1 defect] [2 cooperate]"
            }
        }
    },
    "Codenames": {
        "Spymaster": {
            "command": {
                "description": "The Spymaster gives a one-word clue + number (e.g., '[wind 2]') based on the team's secret words (the clue may not contain any of the words on the board (Codenames Words list)).",
                "sample_action": "[animal 2]"
            }
        },
        "Operative": {
            "command": {
                "description": "The Operative guesses up to N+1 words (e.g., '[breeze]') based on the clue.They can also '[pass]'. Avoid guessing opponent words, neutral words (N), or the Assassin (A), which causes instant loss.",
                "sample_action": "[cat]"
            }
        }
    }
}

class RoundAction(BaseModel):
    game_name: Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"]
    current_round: str
    current_player_name_or_role: str
    action_type: Literal["command", "chat"]


class GameInstructions(BaseModel):
    rule: str
    winning_condition: str


class GameStatus(BaseModel):
    history_rounds: str
    current_rounds: str
    all_team_scores: str


class GameObservation(BaseModel):
    game_name: Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"]
    rule_and_instructions: GameInstructions
    current_player_name_or_role: str


class ObservationParts(BaseModel):
    game_instructions: str
    game_status: str

class StarsAgentTrack2V4(StarsAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_observation_parts(self, observation: str, game_name: str) -> ObservationParts:
        if game_name == "ColonelBlotto":
            split_words = "Win the majority of fields to win the round!"
        elif game_name == "Iterated Prisoner's Dilemma":
            split_words = "The player(s) with the highest score at the end of all rounds wins."
        elif game_name == "Codenames":
            split_words = "4. First team to guess all their words wins."
        else:
            split_words = "\n"
        game_instructions = observation.split(split_words)[0] + split_words
        game_status = observation.split(split_words)[-1]

        return ObservationParts(game_instructions=game_instructions, game_status=game_status)


    def _get_action_instructions(self, observation_parts: ObservationParts) -> RoundAction:
        content = self.generate_with_format(
            prompt=f"""
You are a competitive game player. Make sure you read the game instructions carefully.
Everytime it will show you an overall text observation. The "observation" has its own structure, generally it contains "rule and instructions" which tells you how to play this game.
The other part is "game status", which tells you history rounds and current action target (for first round, it doesn't has history).
 
This is the first part of the observation: \n"{observation_parts.game_instructions}"\n

Now tell me which game are you playing.
Also, in a game, there are different player roles and different action types (e.g. command, chat), there is a reference for you {ACTION_FORMAT}. In this reference, it shows
in each game, for all player roles, display all action descriptions and samples for all action types.  

The second part of the "observation" is \n"{observation_parts.game_status}"\n read it carefully. 
First tell me what current latest round is ! (If there's no round info in the text, respond with "unknown")
Normally in this part, the observation tells you DIRECTLY what's your current action type, DO NOT assume by yourself! and in this part, use content from the REFERENCE to answer me, what's the current round action type?


And reorganize your answers into following structure:
{{
    "game_name": Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"],
    "current_round": "",
    "current_player_name_or_role": "player 0",
    "action_type": Literal["command", "chat"],
}}

""",
            system="Enable deep thinking subroutine.",
            options={"temperature": 0.1},
            output_format=RoundAction.model_json_schema(),
            print_log=True
        )
        print(json.dumps(json.loads(content), indent=2))
        return RoundAction(**json.loads(content))

    def _get_game_observation(self, observation: str) -> GameObservation:
        content = self.generate_with_format(
            prompt=f"""
You are a competitive game player. Make sure you read the game instructions carefully.
This is the game you are playing, everytime it will show you an overall observation, and this is what you've observed now: \n"{observation}"\n

The "observation" has its own structure, generally it contains "rule and instructions" which tells you how to play this game.
The other part is "game status", which tells you history rounds and current action target (for first round, it doesn't has history).
Now, tell me, in first part of the observation, what game are you playing. What is the rule and instruction? What is the name or role of the player you are playing?


And reorganize your answers into following structure:
{{
    "game_name": "game name",
    "rule_and_instructions": {{
        "rule": "game rule",
        "winning_condition": "condition"
    }}
    "current_player_name_or_role": "player 0",
}}
""",
            system="Enable deep thinking subroutine.",
            options={"temperature": 0.1},
            output_format=GameObservation.model_json_schema(),
            print_log=False
        )
        print(json.dumps(json.loads(content), indent=2))
        return GameObservation(**json.loads(content))


    def __call__(self, observation: str) -> str:
        game_observation = self._get_game_observation(observation)
        observation_parts = self._get_observation_parts(observation, game_observation.game_name)
        round_action = self._get_action_instructions(observation_parts)
        return ""


if __name__ == "__main__":

    agent = StarsAgentTrack2V4("cogito:8b")
    # agent = StarsAgentTrack2V4("qwen3:8b")
    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        if game_name == "3-player Iterated Prisoner's Dilemma":
        # if game_name == "Codenames":
        # if game_name == "ColonelBlotto":
            for sample in samples[game_name]:

                print(f"\033[32m{sample}\033[0m")
                print("*" * 300)
                agent(sample)
                print("=" * 300)
                # # break


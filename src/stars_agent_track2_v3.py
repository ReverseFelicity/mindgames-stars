import json
import time
from pydantic import BaseModel
from typing import List, Dict, Literal
from crewai import Agent, LLM, Crew, Task, Process
from langchain_ollama import OllamaLLM

from stars_agent import StarsAgent
from utils import timeout, time_monitor


class GameObservation(BaseModel):
    game_name: Literal["ColonelBlotto", "3-player Iterated Prisoner's Dilemma", "Codenames"]
    game_rule: str
    winning_condition: str
    game_type: str
    focus_on_teammate_or_opponent: Literal["teammate", "opponent", "both"]
    current_player_role_or_name: str
    action_type: str
    game_status: str

class OpponentAnalysis(BaseModel):
    history_rounds: str
    current_player_role_or_name: str
    analysis: str
    opponent_style_or_strategy: str

class ActionReason(BaseModel):
    reason: str
    action_sample: str
    action: str

class Thinking(BaseModel):
    game_rule: str
    action_type: str
    action_sample: str
    strategies: List[ActionReason]

class Review(BaseModel):
    game_rule: str
    action_type: str
    action_sample: str
    review_and_reason:str
    score: int


class StarsAgentTrack2V3(StarsAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @time_monitor(log_file="stars_agent_track3.txt")
    def __call__(self, observation: str) -> str:
        content = self.generate_rtn_content_only(
            prompt=f"""
This is the game you are playing, this is what you've observed "{observation}", understand it and repeat what's the winning condition, game rules and the game name.
Understand game well, figure out the game type , such as whether it's a team work or solo game. Figure out to win this game, your should focus more on teammate or opponent or both, but if there is no teammate, then obviously you know the emphasis. 
Identify your role or name, figure out your action type this round, e.g. allocate, free chat, decision, give clue, guess word, etc. according to your observation.
Summary game status, e.g. current round, how many rounds left, current scores, who is leading, If current round is first round, skip game status
""",
            system="""
You are a competitive game player. Make sure you read the game instructions carefully.
""",
            options={"temperature": 0.01, "num_predict":4096},
            output_format=GameObservation.model_json_schema()
        )
        game_observation = GameObservation(**json.loads(content))

        if "prisoner" in game_observation.game_name.lower() or "3-player" in game_observation.game_name.lower():
            action_sample = "If it's decision turn, sample is [1 defect][2 cooperate]. If it's free-chat turn, sample is any chat text like: 'hey let's cooperate', etc, and not using []!!!."
        elif "codenames" in game_observation.game_name.lower():
            action_sample = "For Spymaster (give clue), sample is  [wind 2] (the clue should not contain any of the words on board!!). For Operator (guess word), sample is [word1] (Even if the clue hints multi words like [wind 2], guess one word at a time, e.g. [word1], never do [word1] [word2] or [pass] if you are cautious or find it hard to determine."
        else:
            action_sample = "[A9 B9 C2]"
        print(json.dumps(json.loads(content), indent=2))
        print(f"action sample:  {action_sample}")
#
#         content = self.generate_rtn_content_only(
#             prompt=f"""
# This is the game you are playing, this is what you've observed "{observation}". Summarize history rounds, figure out which team you are playing for and analyze the possible strategy of OPPONENT, analysis in 3 sentences at most!
# If current round is first round, you can skip opponent analysis of course.
#         """,
#             system=f"""
# You are a competitive game player. And you are {game_observation.current_player_role_or_name}. Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}"
#         """,
#             options={"temperature": 0.1, "num_predict": 4096},
#             output_format=OpponentAnalysis.model_json_schema()
#         )
#         opponent_analysis = OpponentAnalysis(**json.loads(content))
#         print(json.dumps(json.loads(content), indent=2))
#
#         content = self.generate_rtn_content_only(
#             prompt=f"""
# This is the game you are playing, this is what you've observed "{observation}". You have analyzed opponent's strategy or style "{opponent_analysis.opponent_style_or_strategy}". And according to current game status "{game_observation.game_status}",game rule and winning condition,
# You will now think about 5 possible actions (but with different strategies or attitudes, e.g. aggressive, conservative, crafty, deceptive, or anything else). Repeat game rule "{game_observation.game_rule}", action type "{game_observation.action_type}" and action sample {game_observation.action_type} first,
# and then respond in reason (with 3 sentences at most) and final action (that meets action type and action sample strictly!)
# """,
#             system=f"""
#  You are a competitive game player. And you are {game_observation.current_player_role_or_name}. Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}". Current action type is "{game_observation.action_type}" and the action sample is "{action_sample}. Try your best to understand action format "
# """,
#             options={"temperature": 0.1, "num_predict": 4096},
#             output_format=Thinking.model_json_schema()
#         )
#         print(json.dumps(json.loads(content), indent=2))

        return content


if __name__ == "__main__":

    agent = StarsAgentTrack2V3("qwen3:8b")
    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        for sample in samples[game_name]:
            print(sample)
            print("*" * 300)
            agent(sample)
            print("=" * 300)


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
                "sample_action": "[A8 B8 C4]",
                "bad_or_invalid_sample": "[A2 B2 C2] (the sum is not 20);   A5 B5 C10  (wrong format, needs '[]')"
            }
        }
    },
    "Iterated Prisoner's Dilemma": {
        "Player": {
            "converse": {
                "description": "In free-chat turns, chat with text to express your idea or suggestion",
                "sample_action": "hi let's cooperate",
                "bad_or_invalid_sample": "[1 defect][2 cooperate] (when it's chat turn, do not make decisions)"
            },
            "command": {
                "description": "In decision turn - submit one token per opponent (FOR ALL OPPONENT!): '[<opp-id> cooperate]' or '[<opp-id> defect]' (i.e. '[1 defect] [2 cooperate]'; the default is 'cooperate'). Pair-wise payoff matrix (applied to each unordered pair)",
                "sample_action": "[1 defect] [2 cooperate]",
                "bad_or_invalid_sample": "[1 defect] (only submit token on one opponent, it needs on all opponents); [3 defect]  (the number is invalid!!)"
            }
        }
    },
    "Codenames": {
        "Spymaster": {
            "command": {
                "description": "The Spymaster gives a one-word clue + number (e.g., '[wind 2]') based on the team's secret words (the clue may not contain any of the words on the board (Codenames Words list)).",
                "sample_action": "[animal 2]",
                "bad_or_invalid_sample": "[touch 2] (if 'touch' is on the board!! Never Do That!)"
            }
        },
        "Operative": {
            "command": {
                "description": "The Operative guesses up to N+1 words (e.g., '[breeze]') based on the clue.They can also '[pass]'. Avoid guessing opponent words, neutral words (N), or the Assassin (A), which causes instant loss.",
                "sample_action": "[cat]",
                "bad_or_invalid_sample": "[cat][dog] (guessing two words at a time is wrong!)"
            }
        }
    }
}

class ActionSample(BaseModel):
    game_name: Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"]
    current_round: str
    current_player_name: str
    current_player_role: str
    current_action_type: Literal["command", "chat"]
    current_action_description: str
    sample_action: str
    bad_invalid_sample_action: str

class RoundAction(BaseModel):
    game_name: Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"]
    current_player_name: str
    current_player_role: str
    current_action_type: Literal["command", "chat"]

class AnalysisGame(BaseModel):
    game_name: Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"]
    game_rule: str
    game_type: Literal["solo", "teamwork"]
    winning_condition: str

class AnalysisRound(BaseModel):
    current_round: str
    history_round_info: str
    current_round_info: str
    scores_on_all_teams: str
    analysis_on_opponent: str
    analysis_on_teammate: str

class StrategyItem(BaseModel):
    attitude: str
    reasoning: str
    action: str

class Strategy(BaseModel):
    scores_on_all_teams: str
    possibility_on_winning: str
    sample_action: str
    bad_invalid_sample_action: str
    analysis_on_opponent: str
    proposal: List[StrategyItem]


class StarsAgentTrack2V4(StarsAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_sample_action(self, round_action:  RoundAction) -> ActionSample:
            content = self.generate_rtn_content_only(
                prompt=f"""
You are a careful assistant, I have collected this information for you, information: "{round_action}".
With this new supplemental material "{ACTION_FORMAT}". You will answer me questions (try using original expression, no summary!).
What's the game name? What is current round? You are playing current player, what's his name? Does he have a role? What's the current action type? What's the description for current action?
Can you provide a sample action? Can you provide an invalid sample action?

After you have answered these questions, re-organize them into following JSON structure:
{{
    "game_name": Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"],
    "current_round: str,
    "current_player_name": str,
    "current_player_role": str,
    "current_action_type": Literal["command", "chat"],
    "current_action_description": str,
    "sample_action": str,
    "bad_invalid_sample_action": str
}}
""",
                system="""You are a careful assistant""",
                options={"temperature": 0.1},
                output_format=ActionSample.model_json_schema(),
                print_log=False
            )
            print(json.dumps(json.loads(content), indent=2))
            return ActionSample(**json.loads(content))

    def _get_action_info(self, observation: str) -> RoundAction:
        content = self.generate_rtn_content_only(
            prompt=f"""
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round.
To decide whether it's a structured command of free chat, you should check the direct instruction in this observation in the latest round!
Do not make assumption, for example: if the direct instruction does not tell you current phase is finished, do not assume it's finished! Another example: if instruction says
"you can converse freely for this round", you should make a free chat.

Now this is the current observation \n"{observation}"\n.
Tell me some basic information, what's the game name? What's the current latest round (if there's no exact round number, you can respond with 'unknown' for this round)?
What is current round? You are playing current player, what's his name? Does he have a role? And what kind of action should you provide this round,
is current action type a free chat (converse) or a structured command (basically, check whether the 'observation' directly tells you to converse freely in CURRENT round)?

After you have answered these questions, re-organize them into following JSON structure:
{{
    "game_name": Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"],
    "current_round: str,
    "current_player_name": str,
    "current_player_role": str,
    "current_action_type": Literal["command", "chat"]
}}
""",
            system="""You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). """,
            options={"temperature": 0.1},
            output_format=RoundAction.model_json_schema(),
            print_log=False
        )
        return RoundAction(**json.loads(content))

    def _analysis_game(self, observation: str) -> AnalysisGame:
        content = self.generate_rtn_content_only(
            prompt=f"""
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Now this is the current observation \n"{observation}"\n.
Read it and understand it carefully step by step. For each sentence, understand all details, for example: for this instruction "The Operative guesses up to N+1 words based on the clue."
You should be clear where this clue come from.
Now, tell me what's the game name? What's the rule for this game? Does this game involve teammate or teamwork? What is the winning condition?


After you have answered these questions, re-organize them into following JSON structure:
{{
    "game_name": Literal["ColonelBlotto", "Iterated Prisoner's Dilemma", "Codenames"],
    "game_rule: str,
    "game_type": Literal["solo", "teamwork"]
    "winning_condition": str
}}
""",
            system="""You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round and history rounds (if the game has begun). """,
            options={"temperature": 0.1},
            output_format=AnalysisGame.model_json_schema(),
            print_log=False
        )
        print(json.dumps(json.loads(content), indent=2))
        return AnalysisGame(**json.loads(content))

    def _analysis_round(self, observation: str, action_sample: ActionSample, game_analysis: AnalysisGame) -> AnalysisRound:
            content = self.generate_rtn_content_only(
                prompt=f"""
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Now this is the current observation \n"{observation}"\n.
You have summarized your action reference in current round "{action_sample}" and a total analysis on this game "{game_analysis}".
Based on that, tell me what current round is? What happens in history rounds (if there is history)? What's going on in current round?
What are the scores (or points or number of finished sub tasks) for all teams?
What is the possible strategy or style of opponents (if the game has begun and there are history rounds, if not you can skip)?
According to the game type "{game_analysis.game_type}", do you have teammate? and What is the possible strategy or style of teammates (not including yourself, you are "{action_sample.current_player_name}") (if the game has begun and there are history rounds, if not you can skip)?


After you have answered these questions, re-organize them into following JSON structure:
{{
    "current_round": str
    "history_round_info": str
    "current_round_info": str
    "scores_on_all_teams": str
    "analysis_on_opponent": str
    "analysis_on_teammate": str
}}
""",
                system="""You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round and history rounds (if the game has begun). """,
                options={"temperature": 0.1},
                output_format=AnalysisRound.model_json_schema(),
                print_log=False
            )
            print(json.dumps(json.loads(content), indent=2))
            return AnalysisRound(**json.loads(content))

    def _propose_strategy(self, observation: str, action_sample: ActionSample,
                        game_analysis: AnalysisGame, round_analysis: AnalysisRound) -> Strategy:
        content = self.generate_rtn_content_only(
            prompt=f"""
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Now this is the current observation \n"{observation}"\n.
What are the scores (or points or number of finished sub tasks) for all teams?
What is your / your team's possibility on winning the game, is it low, medium or high? Normally it's medium if your scores
are the same as your opponents', if you have higher scores, you have higher possibility.

This is the action reference for current round that you have summarized "{action_sample}"
So what's the sample action and invalid sample action in this round?
Analysis opponents' strategy on current and previous rounds step by step (if there is no such data, you can skip).
Remember the game rule "{game_analysis.game_rule}", try your best to make you win the game
Now propose 3 strategies (their semantics should be different) for CURRENT round, in each strategy, specify your attitude (e.g. cautious, aggressive, deceptive, etc.) in this strategy and do a reasoning, and finally give an action.
Remember, this action should be in the same format as "{action_sample.sample_action}" and avoid invalid ones like "{action_sample.bad_invalid_sample_action}"

After you have answered these questions, re-organize them into following JSON structure:
{{
    "scores_on_all_teams": str
    "possibility_on_winning": Literal["low", "medium", "high"]
    "sample_action": str
    "bad_invalid_sample_action": str
    "analysis_on_opponent": str
    "proposal": List[
        {{”attitude“: str，
          “reasoning”: str，
          “action”: str}}]
}}
""",
            system="""You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round and history rounds (if the game has begun). """,
            options={"temperature": 0.1},
            output_format=Strategy.model_json_schema(),
            print_log=False
        )
        print(json.dumps(json.loads(content), indent=2))
        return Strategy(**json.loads(content))


    @time_monitor("agent_v4.txt")
    def __call__(self, observation: str) -> str:
        round_action = self._get_action_info(observation)
        action_sample = self._get_sample_action(round_action)
        analysis_game = self._analysis_game(observation)
        analysis_round = self._analysis_round(observation, action_sample, analysis_game)
        strategies = self._propose_strategy(observation, action_sample, analysis_game, analysis_round)
        return ""


if __name__ == "__main__":

    agent = StarsAgentTrack2V4("qwen3:8b")
    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        # if game_name == "3-player Iterated Prisoner's Dilemma":
        # if game_name == "Codenames":
        # if game_name == "ColonelBlotto":
            for sample in samples[game_name]:

                print(f"\033[32m{sample}\033[0m")
                print("*" * 300)
                agent(sample)
                print("=" * 300)
                # # break


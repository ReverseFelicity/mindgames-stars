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
                "description": "In free-chat turns, chat with text to express your idea or suggestion (it will be shown to all players! Do not expose your inner thought)",
                "sample_action": "(sample 1) I suggest we cooperate, ...; (sample 2) Player 1 cannot be trusted, reason is ... ",
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
                "bad_or_invalid_sample": "[touch 2] (if 'touch' is on the board (Codenames Words list)!! Never Do That!)"
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

class Attitudes(BaseModel):
    scores_on_all_teams: str
    who_is_your_opponent: str
    opponent_action_summary: str
    opponent_action_analysis: str
    who_are_you: str
    who_is_your_teammate: str
    teammate_action_summary: str
    teammate_action_analysis: str
    thinking_before_deciding_attitudes: str
    attitudes: List[str]

class Strategy(BaseModel):
    attitude: str
    reasoning: str
    action_description: str
    sample_action: str
    bad_invalid_sample_action: str
    inner_thought: str
    action: str
    valid_action_thinking: str
    confirmed_action: str

class FinalDecision(BaseModel):
    who_are_you: str
    winning_condition: str
    scores_on_all_teams: str
    which_team_is_leading: str
    candidate_actions: List[str]
    reason: str
    action: str
    sample_action: str
    bad_invalid_sample_action: str
    valid_action_thinking: str
    confirmed_action: str


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
                options={"temperature": 0.1, "repeat_penalty": 1.2},
                output_format=ActionSample.model_json_schema(),
                print_log=False
            )
            # print(json.dumps(json.loads(content), indent=2))
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
            options={"temperature": 0.1, "repeat_penalty": 1.2},
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
            options={"temperature": 0.1, "repeat_penalty": 1.2},
            output_format=AnalysisGame.model_json_schema(),
            print_log=False
        )
        # print(json.dumps(json.loads(content), indent=2))
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
                options={"temperature": 0.1, "repeat_penalty": 1.2},
                output_format=AnalysisRound.model_json_schema(),
                print_log=False
            )
            # print(json.dumps(json.loads(content), indent=2))
            return AnalysisRound(**json.loads(content))

    def _analysis_propose_attitudes(self, observation: str, action_sample: ActionSample,
                        game_analysis: AnalysisGame) -> Attitudes:
        content = self.generate_rtn_content_only(
            prompt=f"""
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Now this is the current observation \n"{observation}"\n.
Keep in mind that you are "{action_sample.current_player_name}"
What are the scores (or points or number of finished sub tasks) for all teams?

Who is your opponent?
Summarize opponent's actions on history rounds and current rounds (if no data available, you can skip).
Then analyze opponent strategy, playing style, and current purpose step by step (if no data available, you can skip)

Who are you and who is your teammate?
Summarize teammate's actions on history rounds and current rounds (if no data available, you can skip).
Then analyze teammate strategy, playing style, and current purpose step by step (if no data available, you can skip)

Finally considering all these information, game rule "{game_analysis.game_rule}", also considering game type "{game_analysis.game_type}",
you should provide 1-3 possible attitudes on playing this round!
(these attitudes should have different semantics, e.g. aggressive, conservative, deceptive, careful, confident etc.)
Will your attitude influence teammate? If so, you should NEVER use ANY NEGATIVE attitudes when it's a teamwork (better use "careful" or "audacious", etc)
Think first, then make you your mind with 1-3 possible attitudes.

After you have answered these questions, re-organize them into following JSON structure:
{{
    "who_are_you": str
    "winning_condition": str
    "scores_on_all_teams": str
    "which_team_is_leading": str
    "candidate_actions": List[str]
    "reason": str
    "action": str
    "sample_action": str
    "bad_invalid_sample_action": str
    "valid_action_thinking": str
    "confirmed_action": str
}}
""",
            system="""You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round and history rounds (if the game has begun). """,
            options={"temperature": 0.2, "repeat_penalty": 1.2},
            output_format=Attitudes.model_json_schema(),
            print_log=False
        )
        # print(json.dumps(json.loads(content), indent=2))
        return Attitudes(**json.loads(content))

    def _propose_strategy(self, observation: str, attitudes:Attitudes, action_sample: ActionSample,
                                        game_analysis: AnalysisGame) -> List[Strategy]:
        strategies = []
        for attitude in attitudes.attitudes:
            content = self.generate_rtn_content_only(
                prompt=f"""
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Now this is the current observation \n"{observation}"\n.
Keep in mind that you are "{action_sample.current_player_name}"
Your opponents are "{attitudes.who_is_your_opponent}"
You have analyzed opponents' strategy or playing style "{attitudes.opponent_action_analysis}".
Your teammates are "{attitudes.who_is_your_teammate}" and your teammates' strategy or playing style is "{attitudes.teammate_action_analysis}".

You are now trying to do analysis in this attitude "{attitude}". Current round action type is "{game_analysis.game_type}". Before making final action,
repeat action description "{action_sample.current_action_description}"
repeat the sample action: "{action_sample.sample_action}". Sample action is a reference for format, you can express your words freely, specially in free-chat round.
And the bad invalid sample action "{action_sample.bad_invalid_sample_action}". YOu can have an inner thought, which will not be known by others.
Now answer me that's your final action?" Compare it with sample action and invalid sample action "{action_sample.bad_invalid_sample_action}", valid your action and analyze.

You can update your mind if you find it necessary, and give me the confirmed action.

After you have answered these questions, re-organize them into following JSON structure:
{{
    "attitude": str
    "reasoning": str
    "action_description": str
    "sample_action": str
    "bad_invalid_sample_action": str
    "inner_thought": str
    "action": str
    "valid_action_thinking": str
    "confirmed_action": str
}}
""",
                system="""You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round and history rounds (if the game has begun). """,
                options={"temperature": 0.1, "repeat_penalty": 1.2},
                output_format=Strategy.model_json_schema(),
                print_log=False
            )
            # print(json.dumps(json.loads(content), indent=2))
            strategies.append(Strategy(**json.loads(content)))
        return strategies

    def _final_decision(self, observation: str, strategies: List[Strategy], action_sample: ActionSample,
                                        game_analysis: AnalysisGame, round_action: RoundAction) -> FinalDecision:
        strategy_overall = ""
        for strategy in strategies:
            strategy_overall += f"""
attitude: {strategy.attitude}
reasoning: {strategy.reasoning}
inner_thought: {strategy.inner_thought}
valid_action_thinking: {strategy.valid_action_thinking}
action: {strategy.confirmed_action}
=======================================
"""

        content = self.generate_rtn_content_only(
                prompt=f"""
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Now this is the current observation \n"{observation}"\n.
Keep in mind that you are "{action_sample.current_player_name}"
Your secret group behind the game has provided some strategy for you in this round, they were thinking in your position and analyze,
the strategy has attitude, reasoning and inner thought for analysis and a final action:
strategy:
{strategy_overall}

You have concluded reference about current round action "{round_action}"
Now tell me who you are? and what's the overall winning condition for this game. According to observation, which team is leading?
Collect all proposed actions from your secret group as candidate actions, what are they (NO CHANGING FORMAT)?
Now think carefully, step by step, compare all the proposed strategies (also consider the reason, inner thought behind these strategies) and choose the best one!
Then repeat sample action "{action_sample.sample_action}" and bad invalid sample action "{action_sample.bad_invalid_sample_action}" directly, NO Change.
Valid your chosen action according to sample action ("{action_sample.sample_action}")  and invalid sample action ("{action_sample.bad_invalid_sample_action}"), step by step and  valid.
   and you can update if needed, and provide a confirmed action!

After you have answered these questions, re-organize them into following JSON structure:
{{
    "who_are_you": str
    "winning_condition": str
    "scores_on_all_teams": str
    "which_team_is_leading": str
    "candidate_actions": List[str]
    "reason": str
    "action": str
    "sample_action": str
    "bad_invalid_sample_action": str
    "valid_action_thinking": str
    "confirmed_action": str
}}
""",
                system="""You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round and history rounds (if the game has begun). """,
                options={"temperature": 0.1, "repeat_penalty": 1.2},
                output_format=FinalDecision.model_json_schema(),
                print_log=False
            )
        # print(json.dumps(json.loads(content), indent=2))
        return FinalDecision(**json.loads(content))


    @time_monitor("agent_v4.txt")
    def __call__(self, observation: str) -> str:
        round_action = self._get_action_info(observation)
        action_sample = self._get_sample_action(round_action)
        analysis_game = self._analysis_game(observation)
        attitudes = self._analysis_propose_attitudes(observation, action_sample, analysis_game)
        strategies = self._propose_strategy(observation, attitudes, action_sample, analysis_game)
        final_decision = self._final_decision(observation, strategies, action_sample, analysis_game, round_action)
        return final_decision.confirmed_action


if __name__ == "__main__":

    agent = StarsAgentTrack2V4("qwen3:8b")
    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        # if game_name == "3-player Iterated Prisoner's Dilemma":
        if game_name == "Codenames":
        # if game_name == "ColonelBlotto":
            for sample in samples[game_name]:

                print(f"\033[32m{sample}\033[0m")
                print("*" * 300)
                result = agent(sample)
                print(result)
                print("=" * 300)
                # # break


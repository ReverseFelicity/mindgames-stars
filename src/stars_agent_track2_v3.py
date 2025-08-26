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

class GameStatus(BaseModel):
    current_rounds: int
    all_team_scores_or_counts: str

class OpponentAnalysis(BaseModel):
    history_rounds: str
    current_player_role_or_name: str
    analysis: str
    opponent_style_or_strategy: str

class Attitude(BaseModel):
    reason: str
    attitudes: List[str]

class ActionReason(BaseModel):
    reason: str
    action_sample: str
    action: str

class Strategy(BaseModel):
    game_rule: str
    action_type: str
    action_sample: str
    red_lines: str
    attitude: str
    reason: str
    action: str
    do_i_valid_my_answer: bool

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
            options={"temperature": 0.01, "num_predict":4096, "repeat_penalty": 1.1},
            output_format=GameObservation.model_json_schema()
        )
        print(json.dumps(json.loads(content), indent=2))
        game_observation = GameObservation(**json.loads(content))

        content = self.generate_with_format(
            prompt=f"""
This is the game you are playing, this is what you've observed "{observation}", understand it well.
Quickly summarize game status with key information, e.g. current round, current scores, who is leading.
If current round is first round, you can skip game status analysis, respond with 'first round, everyone is equal'. Sometimes the game is not about score, you can summarize with counts according to game definition.
Be short, use 1 sentence only!!! Format: team name : score,   for all team!
    """,
            system="""
You are a competitive game player. Make sure you read the game instructions carefully.
""",
            options={"temperature": 0.01, "num_predict": 4096, "repeat_penalty": 1.1},
            output_format=GameStatus.model_json_schema()
        )
        print(json.dumps(json.loads(content), indent=2))
        game_status = GameStatus(**json.loads(content))


        if "prisoner" in game_observation.game_name.lower() or "3-player" in game_observation.game_name.lower():
            action_sample = "If it's decision turn, sample is [1 defect][2 cooperate]. If it's free-chat turn, sample is any chat text like: 'hey let's cooperate', etc, and not using []!!!."
        elif "codenames" in game_observation.game_name.lower():
            action_sample = "For Spymaster (give clue), sample is  [wind 2] (the clue should not contain any of the words on board (Codenames Words List)!). For Operator (guess word), sample is [word1] (Even if the clue hints multi words like [wind 2], guess one word at a time, e.g. [word1], never do [word1] [word2] or [pass] if you are cautious or find it hard to determine."
        else:
            action_sample = "[A9 B9 C2]"
        print(f"action sample:  {action_sample}")

        content = self.generate_rtn_content_only(
            prompt=f"""
This is the game you are playing, this is what you've observed "{observation}". Summarize history rounds, figure out which team you are playing for and analyze the possible strategy of OPPONENT, analysis in 3 sentences at most!
When it's first round or just starting, you can skip opponent analysis with respond "first round skipping". You have understand that playing this game, between teammate and opponent, priority or emphasis should be placed on {game_observation.focus_on_teammate_or_opponent}.
Since you only have 3 minutes to do action, time is very precious, so if 'teammate' is among your priority, you should pay no attention to your opponent, skip analyzing opponent with respond like 'opponent strategy is not important'. Now start analyzing opponent strategy, (in 3 sentences)
""",
            system=f"""
You are a competitive game player. And you are {game_observation.current_player_role_or_name}. Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}".
You have understand that playing this game, between teammate and opponent, priority or emphasis should be placed on {game_observation.focus_on_teammate_or_opponent}.
        """,
            options={"temperature": 0.1, "repeat_penalty": 1.1},
            output_format=OpponentAnalysis.model_json_schema()
        )
        opponent_analysis = OpponentAnalysis(**json.loads(content))
        print(json.dumps(json.loads(content), indent=2))

        content = self.generate_rtn_content_only(
            prompt=f"""
This is the game you are playing, this is what you've observed "{observation}". Since you only have 3 minutes to do action, time is very precious.
You have understand that playing this game, between teammate and opponent, priority or emphasis should be placed on {game_observation.focus_on_teammate_or_opponent}.
You have analyzed opponent's strategy or style "{opponent_analysis.opponent_style_or_strategy}". And according to current game status "{game_status}",game rule and winning condition: '{game_observation.winning_condition}',
You will now think about 2 possible actions (their semantics should be different). Here are some basic hints, (1) adventurous or aggressive when opponent is leading and they are closer to win. 
(2) confidence, careful, conservative, cautious when your emphasis is on teammate (3) suspicious, conservative or even deceptive on opponent (never on teammate）. 
According to different game status and your priority, try your best to decide. Start with aa short thinking and reasoning (2-3 sentences, only think about attitude, not for action), and decide 2 possible attitude (differ in semantics)
""",
            system=f"""
You are a competitive game player. And you are {game_observation.current_player_role_or_name}. Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}". Current action type is "{game_observation.action_type}".
You have understand that playing this game, between teammate and opponent, priority or emphasis should be placed on {game_observation.focus_on_teammate_or_opponent}.
""",
            options={"temperature": 0.1, "num_predict": 4096, "repeat_penalty": 1.1},
            output_format=Attitude.model_json_schema()
        )
        print(json.dumps(json.loads(content), indent=2))
        attitudes = Attitude(**json.loads(content))

        strategies = """"""
        for attitude in attitudes.attitudes:
            try:
                content = self.generate_with_format(
                    prompt=f"""
This is the game you are playing, this is what you've observed "{observation}". Since you only have 3 minutes to do action, time is very precious.
You have understand that playing this game, between teammate and opponent, priority or emphasis should be placed on {game_observation.focus_on_teammate_or_opponent}.
You have analyzed opponent's strategy or style "{opponent_analysis.opponent_style_or_strategy}" with your analysis "{opponent_analysis.analysis}". And according to current game status "{game_status}",game rule and winning condition: '{game_observation.winning_condition}',
You decide to think and analyze in this attitude and style: "{attitude}". Repeat game rule "{game_observation.game_rule}", action type "{game_observation.action_type}" and action sample {game_observation.action_type} first,
and summarize game red lines from action sample and game rule to KEEP IN MIND!
Start analyzing and reasoning, think step by step, after that propose an action for this round and at last you should valid your action according to rules, red lines and format one by one!!
If you forget to valid your action, there will be consequences! Repeat your final action one more time and make sure you validate it.
""",
                    system=f"""
You are a competitive game player. And you are {game_observation.current_player_role_or_name}. Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}". Current action type is "{game_observation.action_type}".
Current action sample is "{action_sample}". Try your best to understand the action format and keep in mind!
""",
                    options={"temperature": 0.1, "num_predict": 4096, "repeat_penalty": 1.1},
                    output_format=Strategy.model_json_schema()
                )
                print(json.dumps(json.loads(content), indent=2))
                strategy = Strategy(**json.loads(content))
                strategies += f"""
reason: {strategy.reason}
action: {strategy.action}

                """
            except Exception as e:
                print("timeout!!")

        content = self.generate_with_format(
            prompt=f"""
This is the game you are playing, this is what you've observed "{observation}". Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}". Current game status is "{game_status}".
Now your secret team has provided you some strategies with analysis "{strategies}" . Review them first (if it meets format request "{action_sample}" or against game rule "{game_observation.game_rule}"). After that, think carefully step by step and pick
a qualified one from them to be used as the action in this round. If nothing is qualified, then you should provide one by yourself, please.
""",
            system=f"""
You are a competitive game player. And you are {game_observation.current_player_role_or_name}. Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}".
Game rule is "{game_observation.game_rule}" and winning condition is "{game_observation.winning_condition}". Current game status is "{game_status}". Action format: "{action_sample}"
""",
            options={"temperature": 0.1, "repeat_penalty": 1.1},
            output_format=ActionReason.model_json_schema()
        )
        print(json.dumps(json.loads(content), indent=2))
        action_reason = ActionReason(**json.loads(content))

        return action_reason.action


if __name__ == "__main__":

    agent = StarsAgentTrack2V3("qwen3:8b")
    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        if game_name != "Codenames":
            continue
        for sample in samples[game_name]:
            print(sample)
            print("*" * 300)
            agent(sample)
            print("=" * 300)


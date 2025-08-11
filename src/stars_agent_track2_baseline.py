import json
from pydantic import BaseModel
from typing import List, Dict, Literal
from stars_agent import StarsAgent

STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."

class ColonelBlottoAction(BaseModel):
    A: int
    B: int
    C: int

class CodeNamesSPYAction(BaseModel):
    one_word_clue: str
    number: int

class CodeNamesOperativeAction(BaseModel):
    action: str

class ThreePlayerIPDAction(BaseModel):
    opp_id: int
    action: Literal['cooperate', 'defect']


class Player(BaseModel):
    player_type: Literal['colonel_blotto', 'codenames_spy', 'codenames_operative', "three_player_IPD"]


class StarsAgentTrack2BaseLine(StarsAgent):

    def __init__(self, *args, **kwargs):
        if "system_prompt" not in kwargs:
            kwargs["system_prompt"] = STANDARD_GAME_PROMPT
        super().__init__(*args, **kwargs)

    def __call__(self, observation: str) -> str:
        identify_prompt = (f"{observation}\n Upper content is game observation. You should identify which game and which role are you playing."
                           f"If it says about ColonelBlotto, you are player for colonel_blotto. If it says 3-player Iterated Prisoner's Dilemma, you are player for three_player_IPD."
                           f"If it says Codenames, you should notice if you are Spymaster or Operative.")
        thinking, content = self.generate(prompt=identify_prompt, system=self.system_prompt,
                                          output_format=Player.model_json_schema())
        player = Player(**json.loads(content))
        if player.player_type == "colonel_blotto":
            thinking, content = self.generate(prompt=observation, system=self.system_prompt,
                                              output_format=ColonelBlottoAction.model_json_schema())
            solution = ColonelBlottoAction(**json.loads(content))
            output = f"[A{solution.A} B{solution.B} C{solution.C}]"
            while solution.A + solution.B + solution.C != 20:
                prompt = f"{observation}\n My allocation: {observation} But the sum of them is not equal to 20! It should be exactly 20! Update it!"
                thinking, content = self.generate(prompt=prompt, system=self.system_prompt,
                                                  output_format=ColonelBlottoAction.model_json_schema())
                solution = ColonelBlottoAction(**json.loads(content))
                output = f"[A{solution.A} B{solution.B} C{solution.C}]"
            return output
        if player.player_type == "codenames_spy":
            thinking, content = self.generate(prompt=observation, system=self.system_prompt,
                                              output_format=CodeNamesSPYAction.model_json_schema())
            solution = CodeNamesSPYAction(**json.loads(content))
            return f"[{solution.one_word_clue} {solution.number}]"
        if player.player_type == "codenames_operative":
            thinking, content = self.generate(prompt=observation, system=self.system_prompt,
                                              output_format=CodeNamesOperativeAction.model_json_schema())
            solution = CodeNamesOperativeAction(**json.loads(content))
            return f"[{solution.action}]"
        if player.player_type == "three_player_IPD":
            thinking, content = self.generate(prompt=observation, system=self.system_prompt,
                                              output_format=ThreePlayerIPDAction.model_json_schema())
            solution = ThreePlayerIPDAction(**json.loads(content))
            return f"[{solution.opp_id} {solution.action}]"


if __name__ == "__main__":

    agent = StarsAgentTrack2BaseLine("qwen3:8b")
import json
import time
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Literal
from crewai import Agent, LLM, Crew, Task, Process
from langchain_ollama import OllamaLLM
import statistics
from stars_agent import StarsAgent
from utils import timeout, my_logger, time_monitor, extract_python_blocks, run_python_blocks


from typing import Callable, List, Tuple



class StarsAgentTrack2V5(StarsAgent):
    _base_prompt = """
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Here's the observation:
=============================
OBSERVATION_PLACEHOLDER
=============================

When making decisions, you like to use Python to list different cases and do calculation to make a better choice when possible.
You will answer a series of questions before making the final action, for each question, think first then answer, with a fixed format [Thinking] + [Answer], now start:

[Question] Are you human player or LLM player?
[Thinking] I am Qwen3 model, so I am a LLM player
[Answer] I am LLM player
[Question] What's your strength when you play mind games, do you have examples?
[Thinking] I like to use Python to simulate all possible cases
[Answer]  I am good at programming and I can generate Python code to help my decision.
"""
    _question_prompt = """
[Question] QUESTION_PLACEHOLDER
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_chat(self, observation: str) ->str:
        print(f"\033[31m{observation}\033[0m")
        chat_prompt = self._base_prompt.replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            "What game are you playing? What's the rule and winning condition in this game?",
            "What's the role and name of the player you are playing?",
            "Who are your opponents? Do you have teammates and who are they?",
            "Do you know this game? What's your plan on playing this game, is there an optimal strategy?",
            "In this round, you are required to output an action, you don't need to decide this action immediately, but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?"
        ]

        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", question)
            content = self.generate_rtn_content_only(
                prompt=chat_prompt,
                system="/nothink",
                options={
                    "temperature": 0.1, "stop": ["[Question]"], "repeat_penalty": 1.2
                }
            )
            chat_prompt += content.strip()
        return chat_prompt

    def generate_action(self, chat_prompt: str) -> str:
        question_list = [
            "You can use Python to help your decisions, for current around, how would you like to use Python to help your decisions? Provide a good idea, but no code needed",
            "According to your idea in last question, now implement it, write a Python script to help your decision, but avoid printing too many logs, only print key logs"
        ]

        fail_times = 0
        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", question)
            content = self.generate_rtn_content_only(
                prompt=chat_prompt,
                system="/nothink",
                options={
                    "temperature": 0.1, "stop": ["[Question]"], "repeat_penalty": 1.2
                }
            )
            chat_prompt += content.strip()
            blocks = extract_python_blocks(content)
            if len(blocks) > 0:
                code, out, err = run_python_blocks(blocks)
                if code == 0 or fail_times > 2:
                    question_list.append(
                        f"This is the execution result of your code: \n{out}\n Now according to this result, game instructions, action format and our whole chat history, you can try testing it in some basic cases, and see if you need to update your Python code"
                        f"and run more experiments with Python or make the final action right now. So if you need more experiments, provide Python code, or you provide the final action directly"
                    )
                else:
                    fail_times += 1
                    question_list.append(
                        f"This is the execution result of your code, it meets error: \n{err}\n Now think it twice, and update your code"
                    )
        return chat_prompt.split("[Answer]")[-1].strip()

    def valid_action(self, observation: str, action: str) -> str:
        chat_prompt = self._base_prompt.replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            "What game are you playing? What's the rule and winning condition in this game?",
            "What's the role and name of the player you are playing?",
            "Who are your opponents? Do you have teammates and who are they?",
            "Do you know this game? What's your plan on playing this game, is there an optimal strategy?",
            "In this round, you are required to output an action, you don't need to decide this action immediately, but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?",
            f"We have discussed and provide this action to you, '{action}', Please valid it carefully according to game rules (format? constraints? redundant text?, etc.)",
            "Finally, now provide the final answer directly (format is important!)"
        ]

        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", question)
            content = self.generate_rtn_content_only(
                prompt=chat_prompt,
                system="/nothink",
                options={
                    "temperature": 0.1, "stop": ["[Question]"], "repeat_penalty": 1.2
                }
            )
            chat_prompt += content.strip()
        return chat_prompt.split("[Answer]")[-1].strip()

    @time_monitor()
    def __call__(self, observation: str) -> str:
        observation_ = self._observation_wrapper(observation)
        chat_prompt = self.generate_chat(observation_)
        action = self.generate_action(chat_prompt)
        action = self.valid_action(observation_, action)
        return action

if __name__ == "__main__":

    agent = StarsAgentTrack2V5("qwen3:8b")
    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        # if game_name == "3-player Iterated Prisoner's Dilemma":
        # if game_name == "Codenames":
        # if game_name == "ColonelBlotto":
            for sample in samples[game_name]:
                print("*" * 300)
                result = agent(sample)
                print(result)
                print("=" * 300)
                break
            break


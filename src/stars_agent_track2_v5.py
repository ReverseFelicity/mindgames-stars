import json
import time
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Literal
from crewai import Agent, LLM, Crew, Task, Process
import statistics
from stars_agent import StarsAgent
from utils import timeout, my_logger, time_monitor, extract_python_blocks, run_python_blocks
from models import *



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
[Answer]  I am good at programming and I can generate Python code to help my decision."""

    _question_prompt = """
[Question] QUESTION_PLACEHOLDER
"""
    _react_prompt = """
[Thinking] THINKING_PLACEHOLDER
[Answer] ANSWER_PLACEHOLDER
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_to_txt("Hello", mode="w", file_name="StarsAgentTrack2V5")

    def generate_chat(self, observation: str):
        print(f"\033[31m{observation}\033[0m")
        chat_prompt = self._base_prompt.replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            Question(question="What game are you playing? What's the rule and winning condition in this game?"),
            Question(question="What's the role and name of the player you are playing?"),
            Question(question="Who are your opponents? Do you have teammates and who are they?"),
            Question(question="In this round, you are required to output an action, you don't need to decide this action immediately,"
                              " but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?",
                     format_name="ReActWithRound",
                     answer_key_in_format="current_action_type")
        ]
        action_type = "structured command"
        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", str(question.question))
            chat_prompt, blocks, react = self.answer_question(question, chat_prompt)
            if question.format_name == "ReActWithRound":
                action_type = react.answer.current_action_type
        return chat_prompt, action_type

    def answer_question(self, q: Question, chat_prompt: str):
        react = None
        if q.format_name:
            react = self.generate_with_format2(
                prompt=chat_prompt, system="/nothink",
                options={
                    "temperature": 0.1, "stop": ["[Question]"], "repeat_penalty": 1.2
                },
                format_name=q.format_name
            )
            answer = react.answer if isinstance(react.answer, str) else getattr(react.answer, q.answer_key_in_format)
            chat_prompt += self._react_prompt.replace("THINKING_PLACEHOLDER", react.thinking).replace(
                "ANSWER_PLACEHOLDER", answer).strip()
            blocks = extract_python_blocks(answer)
        else:
            content = self.generate_rtn_content_only(
                prompt=chat_prompt, system="/nothink",
                options={
                    "temperature": 0.1, "stop": ["[Question]"], "repeat_penalty": 1.2
                }
            )
            chat_prompt += content.strip()
            blocks = extract_python_blocks(content)
        return chat_prompt, blocks, react

    def generate_action(self, chat_prompt: str) -> str:
        question_list = [
            Question(question="You can use Python to help your decisions, for current around, how would you like to use Python to help your decisions? Provide a good idea, but no code needed"),
            Question(question="According to your idea in last question, now implement it, write a Python script (format: ```python(.*?)```) to help your decision, but avoid printing too many logs, only print key logs")
        ]

        fail_times = 0
        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", str(question.question))
            chat_prompt, blocks, _ = self.answer_question(question, chat_prompt)
            if len(blocks) > 0:
                code, out, err = run_python_blocks(blocks)
                if code == 0 or fail_times > 2:
                    # for code_block in blocks:
                    #     chat_prompt.replace(code_block, "Code Skipped, focus on code result")
                    question_list.append(
                        Question(question=f"This is the execution result of your code: \n'{out}'\n Now according to this result, game instructions, action format and our whole chat history,"
                                          f" you can try testing it in some basic cases, and see if you need to update your Python code"
                                          f" and run more experiments with Python or make the final action right now. So if you need more experiments,"
                                          f" provide Python code(format: ```python(.*?)```), or you provide only 1 final action directly. Remind yourself, this round is a free-chat round or a structured output round")
                    )
                else:
                    fail_times += 1
                    question_list.append(
                        Question(question=f"This is the execution result of your code, it meets error: \n'{err}'\n Now think it twice, and update your code")
                    )
        self._log_to_txt(chat_prompt, "StarsAgentTrack2V5")
        return chat_prompt.split("[Answer]")[-1].strip()

    def valid_action(self, observation: str, action: str) -> str:
        chat_prompt = self._base_prompt.replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            Question(question="What game are you playing? What's the rule and winning condition in this game?"),
            Question(question="What's the role and name of the player you are playing?"),
            Question(question="Who are your opponents? Do you have teammates and who are they?"),
            Question(question="In this round, you are required to output an action, you don't need to decide this action immediately, but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?"),
            Question(question=f"We have discussed and provide this action to you, '{action}', Please valid it carefully according to game rules,"
                              f" and is it free-chat round or structured output round? If it's a free-chat round, is the content in provided 'action' appropriate to others? Do you expose your inner thought accidentally?"),
            Question(
                question=f"Further, about this action, '{action}', please valid it carefully according to game rules, format? constraints? redundant text?, irrelevant symbols like ` etc."),
            Question(question="Finally, according your validation result in last 2 questions, think and provide final action in [Answer]")
        ]

        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", str(question.question))
            chat_prompt, blocks, _ = self.answer_question(question, chat_prompt)
        self._log_to_txt(chat_prompt, "StarsAgentTrack2V5")
        return chat_prompt.split("[Answer]")[-1].strip()

    def output_wrapper(self, action: str):
        if not action.startswith('[') and not action.endswith(']'):
            return f"[{action}]"
        return action

    @time_monitor()
    def __call__(self, observation: str) -> str:
        observation_ = self._observation_wrapper(observation)
        chat_prompt, action_type = self.generate_chat(observation_)
        # action = self.generate_action(chat_prompt)
        # for i in range(2):
        #     action = self.valid_action(observation_, action)
        # self._log_to_txt("\n" + "*" * 200, "StarsAgentTrack2V5")
        # return self.output_wrapper(action)
        return chat_prompt

if __name__ == "__main__":

    agent = StarsAgentTrack2V5("qwen3:8b")

    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        if game_name == "3-player Iterated Prisoner's Dilemma":
        # if game_name == "Codenames":
        # if game_name == "ColonelBlotto":
            for sample in samples[game_name]:
                result = agent(sample)
                print(result)
                print("=" * 300)

import re
import ollama
import json
from ollama import chat, generate, ChatResponse, GenerateResponse
from typing import List, Dict
from utils import time_monitor, my_logger, timeout
from agent import Agent
from datetime import datetime
from models import *


class StarsAgent(Agent):
    memory: list

    def __init__(self, model_name: str="qwen3:8b", think_tags: str = r'<think>.*?</think>', system_prompt: str=None, model_option: dict=None):
        available_models = [model.model for model in ollama.list().models]
        if not model_name in available_models:
            raise Exception(f"model name {model_name} not available, {available_models}")
        self.model_name = model_name
        self.think_tags = think_tags
        self.system_prompt = system_prompt
        self.model_option = model_option if not model_option else {"temperature": 0.2, "num_predict": 2048}
        self.memory = []

    def _log_to_txt(self, content: str, file_name: str="agent.txt", mode: str='a'):
        with open(f"logs/{file_name}.txt", mode, encoding="utf-8") as f:
            f.writelines([
                "\n ========== %s  ==========" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), content
            ])

    def _observation_wrapper(self, observation: str) -> str:
        observation_ = observation.replace(
            "Win the majority of fields to win the round!",
            "Win the majority of fields to win the round (win 2 fields out of 3)!"
        )
        observation_ = observation_.replace(
            "Format: '[A4 B2 C2]'",
            "Format: '[A6 B7 C7]'"
        )
        observation_ = observation_.replace(
            "1 free-chat turns",
            "1 free-chat turns (Format: 'Let us cooperate'. Notice chat will be shared with other players, do not express your inner thought)"
        )
        observation_ = observation_.replace(
            "(the clue may not contain any of the words on the board).",
            ". The clue may not contain any of the words on the board (the Codenames Words list) or you will lose the game instantly!"
        )
        observation_ = observation_.replace(
            "The Operative guesses up to N+1 words (e.g., '[breeze]') based on the clue. They can also '[pass]'.",
            "The Operative guesses up to N+1 words (e.g., '[breeze]') based on the clue. "
            "But Operative should guess 1 word at one time, and there will be N+1 rounds for Operative to guess. They can also '[pass]' to finish guessing. But do not add 'guess'!! Here is sample: [cat]"
        )
        return observation_


    def _split_think_tags(self, origin_text: str):
        match = re.search(self.think_tags, origin_text, flags=re.DOTALL)
        if match:
            thinking_content = match.group(0)
            rest_content = origin_text.replace(thinking_content, '').strip()
            return thinking_content.replace("<think>", "").replace("</think>", "").strip(), rest_content
        else:
            return "", origin_text.strip()

    def generate_with_format(self, prompt: str, output_format: dict, system: str=None, options: dict=None, print_log: bool = False):
        thinking, content = self.generate(prompt=prompt, system=system, print_log=print_log)
        _, content = self.generate(
            prompt=f"""This content contains a Json output: {content}. Extract the Json part and output in following format: {output_format}""", output_format=output_format, print_log=print_log)
        return content

    def generate_with_format2(self, prompt: str, format_name: str, system: str=None, options: dict=None, print_log: bool = False ):
        cl = globals().get(format_name)
        thinking, content = self.generate(prompt=prompt, system=system, print_log=print_log)
        _, content = self.generate(
            prompt=f"""From this content: {content}. According to this following format: {cl.model_json_schema()}, extract content and output in json""", output_format=cl.model_json_schema(), print_log=print_log)
        if print_log:
            print(f"\033[31m{prompt}\033[0m")
            print(f"\033[33m{thinking}\033[0m")
            print(f"\033[34m{content}\033[0m")
        return cl(**json.loads(content))

    def generate_rtn_content_only(self, prompt: str, system: str=None, options: dict=None, output_format=None, print_log=False):
        _, content = self.generate(prompt, system, options, output_format, print_log=print_log)
        return content


    @my_logger("generate.txt")
    def generate(self, prompt: str, system: str=None, options: dict=None, output_format=None, print_log: bool = False):
        if not options: options = self.model_option

        response = generate(model=self.model_name, prompt=prompt, system=system, options=options, format=output_format)
        thinking, content = self._split_think_tags(response.response)
        if print_log:
            print(f"\033[31m{prompt}\033[0m")
            print(f"\033[33m{thinking}\033[0m")
            print(f"\033[34m{content}\033[0m")
        return thinking, content

    def chat(self, messages: List[Dict[str, str]]):
        chat_response: ChatResponse = chat(model=self.model_name, messages=messages)
        content = chat_response['message']['content']
        thinking, content = self._split_think_tags(content)
        return thinking, content

    def __call__(self, observation: str) -> str:
        raise NotImplementedError()


if __name__ == "__main__":

    agent = StarsAgent("cogito:8b")
    thinking, content = agent.generate(prompt="how to cook eggs", system="Enable deep thinking subroutine.")
    print(thinking)
    print("=======")
    print(content)


import re
import ollama
from ollama import chat, generate, ChatResponse, GenerateResponse
from typing import List, Dict
from utils import time_monitor
from agent import Agent


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

    def _split_think_tags(self, origin_text: str):
        match = re.search(self.think_tags, origin_text, flags=re.DOTALL)
        if match:
            thinking_content = match.group(0)
            rest_content = origin_text.replace(thinking_content, '').strip()
            return thinking_content.replace("<think>", "").replace("</think>", "").strip(), rest_content
        else:
            return "", origin_text.strip()

    def generate_with_format(self, prompt: str, output_format: dict, system: str=None, options: dict=None):
        thinking, content = self.generate(prompt=prompt, system=system)
        _, content = self.generate(prompt=f"rewrite this content: '{content}' into target format: {output_format}", system=system, output_format=output_format)
        return content

    def generate_rtn_content_only(self, prompt: str, system: str=None, options: dict=None, output_format=None):
        _, content = self.generate(prompt, system, options, output_format)
        return content

    @time_monitor("generate.txt")
    def generate(self, prompt: str, system: str=None, options: dict=None, output_format=None):
        if not options: options = self.model_option

        response = generate(model=self.model_name, prompt=prompt, system=system, options=options, format=output_format)
        thinking, content = self._split_think_tags(response.response)
        return thinking, content

    def chat(self, messages: List[Dict[str, str]]):
        chat_response: ChatResponse = chat(model=self.model_name, messages=messages)
        content = chat_response['message']['content']
        thinking, content = self._split_think_tags(content)
        return thinking, content

    def __call__(self, observation: str) -> str:
        raise NotImplementedError()


if __name__ == "__main__":

    agent = StarsAgent("qwen3:8b")


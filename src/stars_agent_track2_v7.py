import json

from copy import deepcopy
from stars_agent import StarsAgent
from utils import timeout, time_monitor, extract_python_blocks, run_python_blocks
from models import *
from typing import List



class StarsAgentTrack2V7(StarsAgent):

    _base_prompt = """
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
At the end of each "observation", it will tell you "Please enter the action:", means you should provide a text either is a structured command or a free chat for current round,
and that's according to the game instruction!

Here's the observation:
=============================
OBSERVATION_PLACEHOLDER
=============================
You are good at playing text-based games, because you are good at Python. You like to use Python to list different cases and do calculation to make a better choice when possible.
You also know how to sort result and only list top N ones. You are also familiar with NLTK, which might be useful for semantic cases.
You will answer a series of questions before making the final action, for each question, think first then answer, with a fixed format [Thinking] + [Answer], 

now start:

[Question] Are you human player or LLM player?
[Thinking] I am Qwen3 model, so I am a LLM player
[Answer] I am LLM player
[Question] What's your strength when you play mind games, do you have examples?
[Thinking] I like to use Python to simulate all possible cases
[Answer]  I am good at programming and I can generate Python code to help my decision."""

    _react_prompt = """
[Thinking] THINKING_PLACEHOLDER
[Answer] ANSWER_PLACEHOLDER"""

    _rewrite_prompt="""
Rewrite this following content into new format, extract 'thinking part' and 'answer part' from it:
=============================
REWRITE_PROMPT
=============================
{{
    "thinking": str,
    "answer": str
}}    
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_to_txt("Hello", mode="w", file_name="StarsAgentTrack2V7")
        self._log_to_txt("Hello", mode="w", file_name="generate")

    def _fetch_code_blocks(self, content: str):
        return extract_python_blocks(content)

    def _run_code_blocks(self, code_strs: List[str], code_exe_times=0) -> [int, str, str]:
        code, out, err = run_python_blocks(code_strs)
        if code != 0 and code_exe_times > 3:
            return 0, "Code execution failed too many times. NO Python Code result, please continue", ""
        if len(out.split("\n")) > 50:
            return 1, "Too many log lines, please update your logging logic or processing logic", "Too many log lines, please update your logging logic or processing logic"
        return code, out, err

    def _generate_with_format(self, prompt: str, q: Question, options: dict):
        react = self.generate_with_format2(
            prompt=prompt, system="/nothink",
            options={
                "temperature": 0.1,
                "stop": ["[Question]", "Please enter the action"],
                "repeat_penalty": 1.2
            } if len(options)==0 else options,
            format_name=q.format_name
        )
        return react

    def _add_question_to_prompt(self, prompt: str, q: Question):
        return f"{prompt}\n[Question] {q.question}"

    def _rewrite_thinking_answer(self, thinking_answer, options: dict):
        obj = self._generate_with_format(deepcopy(self._rewrite_prompt).replace("REWRITE_PROMPT", thinking_answer), Question(question="", format_name="ReAct"), options)
        return deepcopy(self._react_prompt).replace("THINKING_PLACEHOLDER", obj.thinking).replace("ANSWER_PLACEHOLDER", obj.answer)

    def _answer_question_without_format(self, prompt: str, options: dict):
        content = self.generate_rtn_content_only(
            prompt=prompt, system="/nothink",
            options={
                "temperature": 0.1,
                "stop": ["[Question]", "Please enter the action"],
                "repeat_penalty": 1.2
            } if len(options)==0 else options
        )
        if not "[Thinking]" in content or not "[Answer]" in content:
            return self._rewrite_thinking_answer(content, options)
        return content


    def _answer_question(self, chat_prompt: str, question: Question, code_exe_times=0, llm_options=None):
        if llm_options is None:
            llm_options = {}
        prompt = self._add_question_to_prompt(chat_prompt, question)
        if not question.format_name:
            thinking_answer = self._answer_question_without_format(prompt, llm_options)
        else:
            obj = self._generate_with_format(prompt, question, llm_options)
            answer = obj.answer if isinstance(obj.answer, str) else getattr(obj.answer, question.answer_key_in_format)
            thinking_answer = deepcopy(self._react_prompt).replace("THINKING_PLACEHOLDER", obj.thinking).replace("ANSWER_PLACEHOLDER", answer)

        code_blocks = self._fetch_code_blocks(thinking_answer)
        observation = None
        if code_blocks:
            code, out, err = self._run_code_blocks(code_blocks, code_exe_times)
            if code == 0:
                observation = out
            else:
                prompt__, thinking_answer__, observation__ = self._answer_question(
                    prompt + thinking_answer, Question(question=f"This is the execution result of your code, it meets error: \n'{err}'\n Now think it twice, and update your code"),
                    code_exe_times+1, llm_options
                )
                thinking_answer = thinking_answer__
                observation = observation__
        thinking_answer_split = thinking_answer.split("[Answer]")
        return prompt, f"{thinking_answer_split[0].strip()}\n[Answer] {thinking_answer_split[1].strip()}", observation


    def get_action(self, observation: str, llm_options=None):
        if llm_options is None:
            llm_options = {}
        base_prompt = deepcopy(self._base_prompt).replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            Question(
                question="What game are you playing? What's the rule and winning condition in this game?"),
            Question(
                question="What's the role and name of the player you are playing?"),
            Question(
                question="In this round, you are required to output an action, you don't need to decide this action immediately,but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?",
                format_name="ReActWithRound", answer_key_in_format="current_action_type"),
            Question(
                question="You can use Python to help your decisions, for current around, how would you like to use Python to help your decisions, also how would you use Python to valid your choice? Provide a good idea, but no code needed"),
            Question(
                question="According to your idea in last question, now implement it, write a Python script (format: ```python(.*?)```) to help your decision, but avoid printing too many logs, only print key logs"),
            Question(
                question="Now according to the above code execution result, considering game instructions, action format and our whole chat history, think twice and provide only 1 final action. Also, remind yourself, this round is a free-chat round or a structured output round"),
        ]
        for q in question_list:
            prompt_with_q, thinking_answer, observation_ = self._answer_question(base_prompt, q, 0, llm_options)
            if observation_ is None:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}"
            else:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}\n[Observation]{observation_}"
        self._log_to_txt(base_prompt, "StarsAgentTrack2V7")
        return base_prompt.split("[Answer]")[-1].strip()

    def valid_action(self, observation: str):
        base_prompt = deepcopy(self._base_prompt).replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            Question(
                question="What game are you playing? What's the rule and winning condition in this game?"),
            Question(
                question="What's the role and name of the player you are playing?"),
            Question(
                question="In this round, you are required to output an action, you don't need to decide this action immediately,but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?",
                format_name="ReActWithRound", answer_key_in_format="current_action_type"),
            Question(
                question="You can use Python to help your decisions, for current around, how would you like to use Python to help your decisions, also how would you use Python to valid your choice? Provide a good idea, but no code needed"),
            Question(
                question="According to your idea in last question, now implement it, write a Python script (format: ```python(.*?)```) to help your decision, but avoid printing too many logs, only print key logs"),
            Question(
                question="Now according to the above code execution result, considering game instructions, action format and our whole chat history, think twice and provide only 1 final action. Also, remind yourself, this round is a free-chat round or a structured output round"),
        ]
        for q in question_list:
            prompt_with_q, thinking_answer, observation_ = self._answer_question(base_prompt, q)
            if observation_ is None:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}"
            else:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}\n[Observation]{observation_}"
        self._log_to_txt(base_prompt, "StarsAgentTrack2V7")
        return base_prompt.split("[Answer]")[-1].strip()

    @time_monitor()
    def __call__(self, observation: str) -> str:
        print(f"\033[31m{observation}\033[0m")
        action1 = self.get_action(observation)
        action2 = self.get_action(observation, {
                "temperature": 0.5,
                "stop": ["[Question]", "Please enter the action"],
                "repeat_penalty": 1.2
            })
        return f"action1: {action1}        action2: {action2}"


if __name__ == "__main__":

    agent = StarsAgentTrack2V7("qwen3:8b")

    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        # if game_name == "3-player Iterated Prisoner's Dilemma":
        # if game_name == "Codenames":
        # if game_name == "ColonelBlotto":
            for sample in samples[game_name]:
                result = agent(sample)
                print(result)
                print("=" * 300)
                # break
            # break

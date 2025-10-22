import json
import time
from copy import deepcopy
import random
from warnings import deprecated

from stars_agent import StarsAgent
from utils import timeout, time_monitor, extract_python_blocks, run_python_blocks, replace_code
from models import *
from typing import List



class StarsAgentTrack2V7(StarsAgent):
    _temperature: float = 0.1
    _validation_prompt = """
Your team are competitive game players, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
For each round, when it's your turn to make an action, you should provide a text either is a structured command or a free chat for current round

Right now, your teammate used some tools and thank carefully, and decide an action for this round, and they also did a validation on it.
You don't the game and rule directly, what you need to do is to compare the proposed action and the validation.
For validation result, it's either reasons for why it fails to meet requirements or a confirmed and repeated action.

this is the proposed action ('=========" is delimiters):
=========
ACTION_PLACEHOLDER
=========

This is your validation result:
=========
VALIDATION_PLACEHOLDER
========

Now you put all these information together into following json format, judge if the validation result means the proposed action is good and valid,
put the reason in 'reasoning', and put the validation result in 'is_action_valid', and at last put the valid action in 'action' (if it's not good, it can be empty)

{{
    "reasoning": your thinking content
    "is_action_valid": bool (if the action is a valid one)
    "action": confirmed action or empty string if it does not meet requirements
}}

"""

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
        if out is None:
            return 1, "No execution result, no logs either", "No execution result, no logs either"
        if len(out.split("\n")) > 50:
            return 1, "Too many log lines, please update your logging logic or processing logic", "Too many log lines, please update your logging logic or processing logic"
        return code, out, err

    def _generate_with_format(self, prompt: str, q: Question, options: dict):
        react = self.generate_with_format2(
            prompt=prompt, system="/nothink",
            options={
                "temperature": self._temperature or 0.1,
                "num_predict": 4096,
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
                "temperature": self._temperature or 0.1,
                "num_predict": 4096,
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
                self._log_to_txt(f"\n************ Origin Code Start ************\n{code_blocks}\n************ Origin Code End ************\n", "StarsAgentTrack2V7")
                thinking_answer = replace_code(thinking_answer)
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

    def get_base_chat_prompt(self, observation: str, llm_options=None):
        if llm_options is None:
            llm_options = {}
        base_prompt = deepcopy(self._base_prompt).replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            Question(
                question="What game are you playing? What's the rule and winning condition in this game?"),
            Question(
                question="What's the role and name of the player you are playing?"),
            Question(
                question="Does the game begin? What round is current round"),
            Question(
                question="Analysis, what should be the structure or format of the 'action' in this round?"),
            Question(
                question="In this round, you are required to output an action, you don't need to decide this action immediately,"
                         "but first make sure it is a command with structured format or a free-chat (if the instruction hints you can converse freely for next 1 round in bottom lines, it means you need to give free-chat right now!)?",
                format_name="ReActWithRound", answer_key_in_format="current_action_type"),
        ]
        for q in question_list:
            prompt_with_q, thinking_answer, observation_ = self._answer_question(base_prompt, q, 0, llm_options)
            if observation_ is None:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}"
            else:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}\n[Observation]{observation_}"
        return base_prompt

    def get_action_by_python(self, chat_prompt: str, additional_questions=None, llm_options=None):
        if additional_questions is None:
            additional_questions = []
        if llm_options is None:
            llm_options = {}
        base_prompt = deepcopy(chat_prompt)
        round_phase = chat_prompt.split("[Answer]")[-1].strip()
        if round_phase == "free-chat":
            question_list = additional_questions + [
                Question(
                    question="Although it's free-chat phase, you can also use Python to help your decisions, for current around, how would you like to use Python to help your decisions, also how would you use Python to valid your choice? Provide a good idea, but no code needed."
                             "You dont need to use Python to get final action directly, you can use Python script to list any information or do experiment, and decide by yourself later."),
                Question(
                    question="According to your idea in last question, now implement it, write a Python script (format: ```python(.*?)```) to help your decision, but avoid printing too many logs, only print key logs, also avoid Emoji, avoid too many comments!"),
                Question(
                    question="Now according to the above code execution result in [Observation], considering game instructions, action format, history rounds and our whole chat history, think twice, put thinking in [Thinking] and provide your words in [Answer]. Your words will be shared with all opponents,  and you know whether to talk frankly, or misleading opponents to realize the simulated effects from your Python code. Mind your language!"),
                Question(
                    question="Since you have just decided the action, think once again to make sure your words do not expose your inner plan, or your inner strategy accidentally!! (It's free-chat phase !! not yet decision phase)"),
            ]

        else:
            question_list = additional_questions + [
                Question(
                    question="You can use Python to help your decisions, for current around, how would you like to use Python to help your decisions, also how would you use Python to valid your choice? Provide a good idea, but no code needed."
                             "You dont need to use Python to get final action directly, you can use Python script to list any information or do experiment, and decide by yourself later."),
                Question(
                    question="According to your idea in last question, now implement it, write a Python script (format: ```python(.*?)```) to help your decision, but avoid printing too many logs, only print key logs, also avoid UnicodeDecodeError"),
                Question(
                    question="Now according to the above code execution result in [Observation], considering game instructions, action format, history rounds and our whole chat history, think twice, put thinking in [Thinking] and provide only 1 final action in [Answer] (basically, if python code is executed successfully, trust code result)"),
            ]


        for q in question_list:
            prompt_with_q, thinking_answer, observation_ = self._answer_question(base_prompt, q, 0, llm_options)
            if observation_ is None:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}"
            else:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}\n[Observation]{observation_}"
        self._log_to_txt(base_prompt, "StarsAgentTrack2V7")
        return  base_prompt.split("[Answer]")[-1].strip()

    def get_action_without_python(self, chat_prompt: str, additional_questions=None, llm_options=None):
        if additional_questions is None:
            additional_questions = []
        if llm_options is None:
            llm_options = {}
        base_prompt = deepcopy(chat_prompt)
        question_list = additional_questions + [
            Question(
                question="Analysis step by step, and you can try the eas"),
            Question(
                question="According to your analysis above, considering game instructions, action format and our whole chat history, think twice and provide only 1 final action in [Answer]"),
        ]
        for q in question_list:
            prompt_with_q, thinking_answer, observation_ = self._answer_question(base_prompt, q, 0, llm_options)
            if observation_ is None:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}"
            else:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}\n[Observation]{observation_}"
        self._log_to_txt(base_prompt, "StarsAgentTrack2V7")
        return  base_prompt.split("[Answer]")[-1].strip()

    def valid_action(self, chat_prompt: str, action: str, llm_options=None):
        if llm_options is None:
            llm_options = {}
        base_prompt = deepcopy(chat_prompt)
        round_phase = chat_prompt.split("[Answer]")[-1].strip()
        if round_phase == "free-chat":
            question_list = [
                Question(
                    question=f"Although it's a free-chat phase, you still used tools and thank carefully and decided to use this action (free-chat text): \n\n{action}\n\nSo you only need to make sure your words do not expose your inner plan accidentally!! (It's free-chat phase !! not yet decision phase)"),
            ]
        else:
            question_list = [
                Question(
                    question="For your current role, what are the red lines? What are the action requirement? And point out the most important one. For example, number constraints? or the words must not contain? or the words must contain.  Only one red line!"),
                Question(
                    question=f"You used tools and thank carefully and decided to use this action: \n\n{action}\n\n You can use Python to help validate this proposed action: {action} with the most important action requirement you have pointed in last question (Only one! the most important one!), provide a Python script (format: ```python(.*?)```) with detailed log printing, print enough logs, but avoid Emoji!"
                             f"Repeat the proposed action in your thinking. Dont you dare assume other proposed action!!"),
                Question(
                    question=f"Now according to the above code execution result in [Observation], think twice, if the action meets requirements, output action in [Answer] directly (which is '{action}')  or list reasons why it does not meet requirements."),
            ]
        for q in question_list:
            prompt_with_q, thinking_answer, observation_ = self._answer_question(base_prompt, q, 0, llm_options)
            if observation_ is None:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}"
            else:
                base_prompt = f"{prompt_with_q}\n{thinking_answer}\n[Observation]{observation_}"
        self._log_to_txt(f"\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n{base_prompt}", "StarsAgentTrack2V7")
        return  base_prompt.split("[Answer]")[-1].strip()

    def get_validation_obj(self, observation: str, action: str, validation: str):
        validation_prompt = (deepcopy(self._validation_prompt)
                             .replace("OBSERVATION_PLACEHOLDER", observation)
                             .replace("ACTION_PLACEHOLDER", action)
                             .replace("VALIDATION_PLACEHOLDER", validation))
        content = self.generate_rtn_content_only(
            prompt=validation_prompt, system="/nothink",
            options={
                "temperature": self._temperature or 0.1, "stop": ["[Question]"], "repeat_penalty": 1.2
            },
            output_format=ReActWithValidation.model_json_schema()
        )
        return ReActWithValidation(**json.loads(content))

    def main_process(self, observation: str):
        meet_requirements = False
        chat_prompt = self.get_base_chat_prompt(observation)
        round_phase = chat_prompt.split("[Answer]")[-1].strip()

        action = ""
        get_action_additions = []
        fail_count = 0
        fail_action_map = {}
        while not meet_requirements:
            action, validation = self.get_action_and_validate(chat_prompt, get_action_additions)
            validation_obj = self.get_validation_obj(observation, action, validation)
            print(validation_obj)
            if validation_obj.is_action_valid:
                meet_requirements = True
                action = validation_obj.action
            else:
                fail_count += 1
                if action not in fail_action_map:
                    fail_action_map[action] = 0
                fail_action_map[action] += 1
                if fail_count > 3:
                    action = max(fail_action_map, key=fail_action_map.get)
                    meet_requirements = True
                else:
                    get_action_additions.append(Question(question=f"You tried this action for this round '{action}', but it doesn't meet requirements because of '{validation_obj.reasoning}', Learn the lesson!! Dont make the same mistake!! Now answer me what lesson have you learnt"))
        if round_phase != "free-chat":
            action = self.output_wrapper(action)
        return action


    def get_action_and_validate(self, chat_prompt: str, additional_questions=None, llm_options=None):
        action = self.get_action_by_python(chat_prompt, additional_questions, llm_options)
        validation = self.valid_action(chat_prompt, action, llm_options)
        return action, validation

    def output_wrapper(self, action: str):
        if not action.startswith('[') and not action.endswith(']'):
            return f"[{action}]"
        return action


    def _get_one_action(self, observation: str, temperature: float):
        self._temperature = temperature
        s = time.time()
        action = self.main_process(observation)
        return action, time.time()-s

    @time_monitor()
    def __call__(self, observation: str) -> str:
        observation = self._observation_wrapper(observation)
        print(f"\033[31m{observation}\033[0m")
        actions = []
        action1, time1 = self._get_one_action(observation, 0.1)
        actions.append(action1)
        # if time1 < 60:
        #     action2, time2 = self._get_one_action(observation, 0.2)
        #     actions.append(action2)

        self._log_to_txt("\n"+"*"*300+"\n", "StarsAgentTrack2V7")
        # print(" | ".join(actions))
        return random.choice(actions)


if __name__ == "__main__":

    agent = StarsAgentTrack2V7("qwen3:8b")

    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        # if game_name == "3-player Iterated Prisoner's Dilemma":
        # if game_name == "Codenames":
        if game_name == "ColonelBlotto":
            for sample in samples[game_name]:
                result = agent(sample)
                print(result)
                print("*" * 300)
            break

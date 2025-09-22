import json

from copy import deepcopy
from stars_agent import StarsAgent
from utils import timeout, time_monitor, extract_python_blocks, run_python_blocks
from models import *



class StarsAgentTrack2V6(StarsAgent):

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

    _question_prompt = """
[Question] QUESTION_PLACEHOLDER
"""
    _react_prompt = """
[Thinking] THINKING_PLACEHOLDER
[Answer] ANSWER_PLACEHOLDER
"""

    _wrong_format_rewrite_prompt = """
You are a competitive game player, You are playing a game based on text, and the text contains all game observation with rules, instructions, current round
and history rounds (if the game has begun). This text is called "observation".
Right now, you are generating this content:

=============================
CONTENT_PLACEHOLDER
=============================

it does not meet the format, so please rewrite it into two parts, (1) Thinking (2) Answer. Figure out which part is thinking, and which part is as answer

"""


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_to_txt("Hello", mode="w", file_name="StarsAgentTrack2V6")
        self._log_to_txt("Hello", mode="w", file_name="generate")

    def generate_chat(self, observation: str):
        print(f"\033[31m{observation}\033[0m")
        chat_prompt = self._base_prompt.replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            Question(question="What game are you playing? What's the rule and winning condition in this game?"),
            Question(question="What's the role and name of the player you are playing?"),
            Question(question="In this round, you are required to output an action, you don't need to decide this action immediately,"
                              " but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?",
                     format_name="ReActWithRound",
                     answer_key_in_format="current_action_type")
        ]

        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", str(question.question))
            chat_prompt, blocks = self.answer_question(question, chat_prompt)

        return chat_prompt, chat_prompt.split("[Answer]")[-1].strip()


    def answer_question_with_format(self, q: Question, chat_prompt:str):
        chat_prompt_ = deepcopy(chat_prompt)

        react = self.generate_with_format2(
            prompt=chat_prompt_, system="/nothink",
            options={
                "temperature": 0.1, "stop": ["[Question]", "Please enter the action"], "repeat_penalty": 1.2
            },
            format_name=q.format_name
        )
        answer = react.answer if isinstance(react.answer, str) else getattr(react.answer, q.answer_key_in_format)
        chat_prompt_ += self._react_prompt.replace("THINKING_PLACEHOLDER", react.thinking).replace(
            "ANSWER_PLACEHOLDER", answer).strip()
        blocks = extract_python_blocks(answer)
        return chat_prompt_, blocks

    def answer_question_without_format(self, q: Question, chat_prompt:str):
        chat_prompt_ = deepcopy(chat_prompt)
        content = self.generate_rtn_content_only(
            prompt=chat_prompt, system="/nothink",
            options={
                "temperature": 0.1, "stop": ["[Question]"], "repeat_penalty": 1.2
            }
        )
        if not "[Thinking]" in content or not "[Answer]" in content:
            chat_prompt_, blocks = self.answer_question_with_format(Question(question="", format_name="ReAct"),
                                             chat_prompt=self._wrong_format_rewrite_prompt.replace("CONTENT_PLACEHOLDER", content))
        else:
            chat_prompt_ += content.strip()
            blocks = extract_python_blocks(content)
        return chat_prompt_, blocks

    def answer_question(self, q: Question, chat_prompt: str):
        if q.format_name:
            return self.answer_question_with_format(q, chat_prompt)
        else:
            return self.answer_question_without_format(q, chat_prompt)


    def analysis_code(self, code, out, err, fail_times):
        if fail_times > 3:
            return True, out or "No Execution Result"
        elif code == 0:
            if len(out.split("\n")) > 20:
                return False, "Too many logs  printing (more than 20 lines)"
            else:
                return True, out
        else:
            return False, err

    @timeout(80)
    def generate_action(self, chat_prompt: str) -> str:
        question_list = [
            Question(question="You can use Python to help your decisions, for current around, how would you like to use Python to help your decisions, also how would you use Python to valid your choice? Provide a good idea, but no code needed"),
            Question(question="According to your idea in last question, now implement it, write a Python script (format: ```python(.*?)```) to help your decision, but avoid printing too many logs, only print key logs")
        ]

        fail_times = 0
        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", str(question.question))
            chat_prompt, blocks = self.answer_question(question, chat_prompt)
            if len(blocks) > 0:
                code, out, err = run_python_blocks(blocks)
                finished, content = self.analysis_code(code, out, err, fail_times)

                if finished:
                    question_list.append(
                        Question(question=f"This is the execution result of your code: \n'{out}'\n. You trust your code and the result. Now according to this result, game instructions, action format and our whole chat history,"
                                          f" you can try testing it in some basic cases, and see if you need to update your Python code"
                                          f" and run more experiments with Python or make the final action right now. So if you need more experiments,"
                                          f" provide Python code(format: ```python(.*?)```), or you provide only 1 final action directly!! Remind yourself, this round is a free-chat round or a structured output round")
                    )
                else:
                    fail_times += 1
                    question_list.append(
                        Question(question=f"This is the execution result of your code, it meets error: \n'{err}'\n Now think it twice, and update your code")
                    )
        self._log_to_txt(chat_prompt, "StarsAgentTrack2V6")
        return chat_prompt.split("[Answer]")[-1].strip()

    def valid_action(self, observation: str, action: str) -> str:
        chat_prompt = self._base_prompt.replace("OBSERVATION_PLACEHOLDER", observation)
        question_list = [
            Question(question="What game are you playing? What's the rule and winning condition in this game?"),
            Question(question="What's the role and name of the player you are playing?"),
            Question(question="In this round, you are required to output an action, you don't need to decide this action immediately, but first make sure is it a command with structured format or a free-chat (if the instruction hints you can converse freely) ?"),
            Question(question=f"We have discussed and provide this action to you, '{action}', Please valid it carefully according to game rules,"
                              f" and is it free-chat round or structured output round? If it's a free-chat round, is the content in provided 'action' appropriate to others? Do you expose your inner thought accidentally?"),
            Question(
                question=f"Further, about this action, '{action}', please valid it carefully according to game rules, format? constraints? redundant text?, irrelevant symbols like ` etc. Is there any text that cannot exist in the action but you accidentally use??"),
            Question(question="Finally, according your validation result in last 2 questions, think and provide final action in [Answer]")
        ]

        for question in question_list:
            chat_prompt += self._question_prompt.replace("QUESTION_PLACEHOLDER", str(question.question))
            chat_prompt, blocks  = self.answer_question(question, chat_prompt)
        self._log_to_txt(chat_prompt, "StarsAgentTrack2V6")
        return chat_prompt.split("[Answer]")[-1].strip()

    def output_wrapper(self, action: str):
        if not action.startswith('[') and not action.endswith(']'):
            return f"[{action}]"
        return action

    @time_monitor()
    def __call__(self, observation: str) -> str:
        observation_ = self._observation_wrapper(observation)
        chat_prompt, action_type = self.generate_chat(observation_)
        try:
            action = self.generate_action(chat_prompt)
        except BaseException as e:
            print(e)
            action = "Timeout, fail to generate an action"
        for i in range(1):
            action = self.valid_action(observation_, action)
        self._log_to_txt("\n" + "*" * 200, "StarsAgentTrack2V6")
        return self.output_wrapper(action)


if __name__ == "__main__":

    agent = StarsAgentTrack2V6("qwen3:8b")

    with open("samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    for game_name in samples:
        # if game_name == "3-player Iterated Prisoner's Dilemma":
        # if game_name == "Codenames":
        if game_name == "ColonelBlotto":
            for sample in samples[game_name]:
                result = agent(sample)
                print(result)
                print("=" * 300)
                break
            break

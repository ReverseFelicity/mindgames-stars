from typing import Literal
from pydantic import BaseModel

class ReAct(BaseModel):
    thinking: str
    answer: str

class Question(BaseModel):
    question: str = ""
    format_name: str = None
    answer_key_in_format: str = None

class Round(BaseModel):
    current_action_type: Literal["free-chat", "structured command"]

class ReActWithRound(BaseModel):
    thinking: str
    answer: Round
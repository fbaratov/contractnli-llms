from pydantic import BaseModel
from typing import Literal

class BinaryNLIResponse(BaseModel):
    classification: Literal["Contradiction", "Entailment"]
    thinking: str

class BinaryNLIReversed(BaseModel):
    explanation: str
    classification: Literal["Contradiction", "Entailment"]

class NLIResponse(BaseModel):
    explanation: str
    classification: Literal["Contradiction", "Entailment", "NotMentioned"]

class NLIClassification(BaseModel):
    classification: Literal["Contradiction", "Entailment", "NotMentioned"]

class NLI4WillsResponse(BaseModel):
    explanation: str
    classification: Literal["support", "refute", "unrelated"]
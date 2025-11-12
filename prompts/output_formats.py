from pydantic import BaseModel
from typing import Literal

class BinaryNLIResponse(BaseModel):
    classification: Literal["Contradiction", "Entailment"]
    thinking: str

class BinaryNLIReversed(BaseModel):
    explanation: str
    classification: Literal["Contradiction", "Entailment"]

class NLIResponse(BaseModel):
    classification: Literal["Contradiction", "Entailment", "NotMentioned"]
    thinking: str

class EvidenceResponse(BaseModel):
    evidence: list[str]
    thinking: str
    
class JointResponse(BaseModel):
    classification: str
    evidence: list[str]
    thinking: str
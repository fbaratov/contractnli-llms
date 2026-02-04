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

class DocNLI(BaseModel):
    explanation: str
    classification: Literal["entailment", "not_entailment"]

class NLI4WillsResponse(BaseModel):
    explanation: str
    classification: Literal["support", "refute", "unrelated"]

class EvidenceResponse(BaseModel):
    evidence: list[str]
    thinking: str
    
class JointResponse(BaseModel):
    classification: str
    evidence: list[str]
    thinking: str
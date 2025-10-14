from pydantic import BaseModel

class NLIResponse(BaseModel):
    classification: str
    thinking: str

class EvidenceResponse(BaseModel):
    evidence: list[str]
    thinking: str
    
class JointResponse(BaseModel):
    classification: str
    evidence: list[str]
    thinking: str
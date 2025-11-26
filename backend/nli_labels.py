import enum

class ExNLILabel(enum.Enum):
    NOT_MENTIONED = 0
    ENTAILMENT = 1
    CONTRADICTION = 2
    INVALID_ANSWER = 3 # LLM moment

    @classmethod
    def from_str(cls, s: str):
        if s == 'NotMentioned':
            return cls.NOT_MENTIONED
        elif s == 'Entailment':
            return cls.ENTAILMENT
        elif s == 'Contradiction':
            return cls.CONTRADICTION
        elif s == 'InvalidAnswer' or s == None:
            return cls.INVALID_ANSWER
        else:
            raise ValueError(f'Invalid input "{s}" to ExNLILabel.from_str.')

    def to_anno_name(self):
        if self == ExNLILabel.NOT_MENTIONED:
            return 'NotMentioned'
        elif self == ExNLILabel.ENTAILMENT:
            return 'Entailment'
        elif self == ExNLILabel.CONTRADICTION:
            return 'Contradiction'
        elif self == ExNLILabel.INVALID_ANSWER:
            return 'InvalidAnswer'
        else:
            print(self)
            assert not 'Should not get here'
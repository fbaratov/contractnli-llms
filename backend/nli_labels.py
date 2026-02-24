import enum



class ExNLILabel(enum.Enum):
    # this is literally an umbrella class which includes everything possible so that i don't have to make it more complex. add new enums here to add new datasets. clean? no. simple? yes.
    # using original labels from datasets, as label names will have semantic value for LLM reasoning as opposed to just being aliases.
    
    ### ContractNLI
    NOT_MENTIONED = 0
    ENTAILMENT = 1
    CONTRADICTION = 2

    ### LLM Expansion pack
    INVALID_ANSWER = 3 # LLM moment

    ### NLI4Wills addition
    REFUTE = 4
    SUPPORT = 5
    UNRELATED = 6

    @classmethod
    def from_str(cls, s: str):
        match s:
        
            # ContractNLI
            case 'NotMentioned':
                return cls.NOT_MENTIONED
            case 'Entailment':
                return cls.ENTAILMENT
            case 'Contradiction':
                return cls.CONTRADICTION
            
            # LLM addition
            case 'InvalidAnswer':
                return cls.INVALID_ANSWER
            case None:
                return cls.INVALID_ANSWER
            
            # NLI4Wills
            case 'support':
                return cls.SUPPORT
            case 'refute':
                return cls.REFUTE
            case 'unrelated':
                return cls.UNRELATED

            # invalid label case
            case _:
                raise ValueError(f'Invalid input "{s}" to ExNLILabel.from_str.')

    def to_anno_name(self):
        match self:
        
            # ContractNLI
            case ExNLILabel.NOT_MENTIONED:
                return 'NotMentioned'
            case ExNLILabel.ENTAILMENT:
                return 'Entailment'
            case ExNLILabel.CONTRADICTION:
                return 'Contradiction'
            
            # LLM
            case ExNLILabel.INVALID_ANSWER:
                return 'InvalidAnswer'
            
            # NLI4Wills
            case ExNLILabel.SUPPORT:
                return "support"
            case ExNLILabel.REFUTE:
                return "refute"
            case ExNLILabel.UNRELATED:
                return "unrelated"

            # bad label
            case _:
                raise ValueError(f'Invalid input "{cls}" to ExNLILabel.to_anno_name.')
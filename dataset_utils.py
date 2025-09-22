from contract_nli.dataset.loader import ContractNLIExample
import json

def get_evidence(example: ContractNLIExample):
    evidence = []
    for i in example.annotated_spans:
        span = example.spans[i]
        evidence.append(example.context_text[span[0]:span[1]])
    return evidence

def load_dataset(dset_path: str):
    with open(dset_path) as fin:
        input_dict = json.load(fin)
    examples = ContractNLIExample.load(input_dict)
    return examples


if __name__=="__main__":
    dev_dataset_path = 'data/dev.json'
    dset = load_dataset(dev_dataset_path)
    
    for ex in dset:
        evidence = get_evidence(ex)
        print(evidence)
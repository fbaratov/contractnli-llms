from backend.dataset.loader import AbridgedDocNLILoader, ContractNLILoader, DocNLILoader, NLI4WillsLoader
from ..contract_nli_bert.contract_nli.dataset.loader import ContractNLIExample
import json

def get_evidence(example: ContractNLIExample):
    evidence = []
    for i in example.annotated_spans:
        span = example.spans[i]
        evidence.append(example.context_text[span[0]:span[1]])
    return evidence

def load_dataset(dset_path: str, dset_type):
    dset_type = dset_type.lower()
    match dset_type:
        case "contract_nli":
            with open(dset_path) as fin:
                input_dict = json.load(fin)
            examples = ContractNLIExample.load(input_dict)
            loader = ContractNLILoader(examples)
            return loader
        case "docnli":
            examples = DocNLILoader(dset_path)
            return examples
        case "abridged_docnli":
            examples = AbridgedDocNLILoader(dset_path)
            return examples
        case "nli4wills":
            examples = NLI4WillsLoader(dset_path)
            return examples
        case _:
            raise ValueError(f"{dset_type} not a valid dataset!")

def retrieve_annotation(dataset: dict, document: int, hypothesis: str) -> dict:
    return dataset[document]["annotation_sets"][0]["annotations"][hypothesis]

def organize_dataset(list_dataset):
    dict_dataset = {}
    for d in list_dataset["documents"]:
        dict_dataset[d["id"]] = d

    return dict_dataset


if __name__=="__main__":
    dev_dataset_path = 'data/dev.json'
    dset = load_dataset(dev_dataset_path)
    
    for ex in dset:
        evidence = get_evidence(ex)
        print(evidence)
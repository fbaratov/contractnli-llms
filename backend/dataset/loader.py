from backend.format_json import encode_dict, encode_onehot_vector
from backend.utils import load_json
import csv
from ..nli_labels import ExNLILabel


class DocNLIExample:
    """
    A single training/test example for NLI. Designed to mimic ContractNLIExample functions but accept the DocNLI dataset.
    Evidence identification task items are practically gone.
    
    Args:
        data_id: The example's unique identifier
        hypothesis_text: The hypothesis string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
    """

    def __init__(
        self,
        hypothesis_text,
        context_text,
        document_id,
        label,
    ):
        self.context_text = context_text
        self.hypothesis_text: str = hypothesis_text
        self.label = label
        self.document_id = document_id
        self.hypothesis_id = str(document_id) # just make them the same but make this a str for safety/consistency :)

class NLILoader:
    def __init__(self):
        self.data = []
    
    def __call__(self, sample_id):
        return self.data[sample_id]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class ContractNLILoader(NLILoader):
    """
    Simple wrapper to basically put contractnli list and class names into one class.
    """

    def __init__(self, examples, class_names=["Entailment", "Contradiction", "NotMentioned"]):
        self.data = examples
        self.class_names = class_names


class DocNLILoader(NLILoader):
    def __init__(self, dset_path):
        json_data = load_json(dset_path)
        self.data = self.convert_data(json_data)

    def convert_data(self, json_data):
        ex_data = []
        for i, d in enumerate(json_data):
            nli_example = DocNLIExample(
                context_text=d["premise"],
                hypothesis_text=d["hypothesis"],
                label=d["label"],
                document_id=i
            )
            ex_data.append(nli_example)

        return ex_data


class AbridgedDocNLILoader(DocNLILoader):
    
    def convert_data(self, json_data):
        ph_track = {}
        ex_data = []
        for i, d in enumerate(json_data):
            premise = d["premise"]
            hypothesis = d["hypothesis"]
            label = d["label"]

            if premise not in ph_track.keys():
                ph_track[premise] = []


            # save one label per premise/label pair to reduce test set size
            if label in ph_track[premise]:
                continue

            nli_example = DocNLIExample(
                context_text=d["premise"],
                hypothesis_text=d["hypothesis"],
                label=d["label"],
                document_id=i
            )
            ex_data.append(nli_example)

            ph_track[premise].append(label)
        return ex_data
    
class NLI4WillsExample:
    """
    A single training/test example for NLI. Designed to mimic ContractNLIExample functions but accept the NLI4Wills dataset.
    Evidence identification task items are practically gone.
    
    Args:
        data_id: The example's unique identifier
        hypothesis_text: The hypothesis string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
    """

    def __init__(
        self,
        statement,
        law,
        conditions,
        document_id,
        statement_id,
        label,
    ):
        self.statement_text = statement
        self.law_text: str = law
        self.conditions_text = conditions
        self.label = label
        self.document_id = document_id
        self.hypothesis_id = str(document_id) # just make them the same but make this a str for safety/consistency :)
        self.law_id = document_id # not necessary to save since document_id is already unique per sample but oh well :))
        self.statement_id = statement_id

class NLI4WillsLoader(NLILoader):
    def __init__(self, dset_path):
        self.class_names = [ExNLILabel.to_anno_name(ExNLILabel.REFUTE),
                       ExNLILabel.to_anno_name(ExNLILabel.SUPPORT),
                       ExNLILabel.to_anno_name(ExNLILabel.UNRELATED),
                       ExNLILabel.to_anno_name(ExNLILabel.INVALID_ANSWER)]
        self.data = self.load_from_csv(dset_path)


    def load_from_csv(self, dset_path) -> list[NLI4WillsExample]:
        data = []

        with open(dset_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, row in enumerate(reader):
                if i == 0: # skip header row
                    continue

                id, statement, conditions, law, classification = row

                label = self.label_int2str(int(classification))

                example = NLI4WillsExample(
                    document_id=i, # use index instead of doc_id to make things simpler wrt laws and hypotheses
                    statement=statement,
                    law=law,
                    statement_id = id, # save just in case
                    conditions=conditions,
                    label=label,
                )
                data.append(example)

        return data
            
    def label_str2int(self, label_str):
        return self.class_names.index(label_str)

    def label_int2str(self, label_int):
        return self.class_names[label_int]
    
    def extract_predictions(self):
        preds = []
        for example in self.data:
            label = example.label
            preds.append(self.label_str2int(label))
        return preds
    
    @classmethod
    def label_cnli(self, label_nli4wills):
        
        # convert before using
        if type(label_nli4wills) is str:
            label_nli4wills = ExNLILabel.from_str(label_nli4wills)

        match label_nli4wills:
            case ExNLILabel.SUPPORT:
                return ExNLILabel.ENTAILMENT
            case ExNLILabel.REFUTE:
                return ExNLILabel.CONTRADICTION
            case ExNLILabel.UNRELATED:
                return ExNLILabel.NOT_MENTIONED
            case ExNLILabel.INVALID_ANSWER:
                return ExNLILabel.INVALID_ANSWER
            case _:
                raise ValueError(f"Bad value for label {label_nli4wills} in nli4wills operation")
            
    def to_cnli(self) -> dict[str, list]:
        cnli_data = {
            "documents": [],
            "labels": {}
        }
        for example in self.data:
            cnli_label = self.label_cnli(example.label)

            cnli_example = {
                "id": example.document_id,
                "annotation_sets": [
                    {
                        "annotations": {
                            example.hypothesis_id : {
                                "choice": ExNLILabel.to_anno_name(cnli_label)
                            }
                        }
                    }
                ]
            }

            cnli_data["documents"].append(cnli_example)
            cnli_data["labels"][example.hypothesis_id] = {
                    "short_description": "Placeholder",
                    "hypothesis": example.statement_text
                }
        return cnli_data
    
    def output_to_cnli(self, output):
        # changes labels to be equivalent to contractnli dataset
        annotations = output["annotation_sets"][0]["annotations"]

        # there's only one, but do this anyway for future robustness
        for hypothesis_id in annotations.keys():
            prediction = annotations[hypothesis_id]["prediction"]
            pred_cnli = self.label_cnli(prediction)
            choice = encode_onehot_vector(pred_cnli)[:4]
            
            class_probs = encode_dict(choice)
            annotations[hypothesis_id]["class_probs"] = class_probs
        
        return output
    
    def extract_predictions_from_response(self, response):
        int_preds = []
        annotations = response["annotation_sets"][0]["annotations"]
        
        # runs only once but done to futureproof more or less :)))
        for k, v in annotations.items():
            prediction = v["prediction"]
            int_preds.append(self.label_str2int(prediction))

        return int_preds
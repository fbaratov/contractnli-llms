import sys

from tqdm import tqdm

from backend.contract_nli_bert.contract_nli.dataset.loader import NLILabel
from backend.dataset.utils import load_dataset
from backend.format_json import save_response

class EmptyOutput:
    def __init__(self):
        self.thinking = None
        self.response = None
        self.logprobs = None

        
def majority_labels(dataset, prediction, output_dir):

    #hardcode a config
    config = {}
    config["model"] = f"majority_baseline_{prediction}"
    config["options"] = None
    config["prompt"] = None
    config["binary"] = False

    evidence = None
    output = None

    for i, ex in enumerate(tqdm(dataset)):
        # if i % 10 > 0:
        #     continue

        if config["binary"] and ex.label == NLILabel.NOT_MENTIONED:
            continue

        # save answer to output/sample_idx
        save_response(config, ex, prediction, evidence, EmptyOutput(), output_dir, inf_time=None)


if __name__=="__main__":
    dset_path, prediction, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    dataset = load_dataset(dset_path=dset_path, dset_type="contract_nli")

    majority_labels(dataset, prediction, output_dir)

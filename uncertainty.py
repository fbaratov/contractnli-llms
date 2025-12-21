import click
import os
from backend.utils import load_json, load_response_dict
from backend.dataset_utils import organize_dataset
from backend.uncertainty.uncertainty import remove_invalid, calculate_probs, sort_correct_wrong, boxplot_analysis
from backend.uncertainty.metrics import metrics_from_list

@click.command()
@click.option("--dataset", type=click.Path(exists=True), default="data/test.json")
@click.option("--out_file", type=click.Path())
@click.option("--structure", type=str, default=None)
@click.option("--response_dir", type=click.Path(exists=True), multiple=True)
def main(dataset, out_file, response_dir, structure):
    # setup out dir
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # load dataset
    dataset = load_json(dataset)
    dataset = organize_dataset(dataset)

    # load and process responses
    correct, wrong = {}, {}
    for fp in response_dir:
        responses = load_response_dict(fp)
        remove_invalid(responses)

        # add probabilities to responses
        calculate_probs(responses, structure=structure)

        # find correct/wrong
        c, w = sort_correct_wrong(dataset, responses)

        # complicated indexing to save on boilerplating 
        for big_dict, sub_dict in [(correct, c), (wrong, w)]:
            if len(big_dict) == 0:
                for k,v in sub_dict.items():
                    big_dict[k] = v
            else:
                for k,v in big_dict.items():
                    big_dict[k] = v + sub_dict[k]

    
    ### conduct analyses

    # do boxplots for classifications per response/seed
    boxplot_analysis(correct, wrong, out_file)

    # do sentence confidence analysis for INCORRECT answers, format this in a nice way.
    # sentence_analysis()


if __name__=="__main__":
    main()
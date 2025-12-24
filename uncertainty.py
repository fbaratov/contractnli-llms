import click
import os
from backend.utils import load_json, load_response_dict
from backend.dataset_utils import organize_dataset
from backend.uncertainty.utils import remove_invalid, calculate_probs, sort_correct_wrong, find_relevant_logprobs
from backend.uncertainty.visualization import boxplot_analysis
from backend.uncertainty.metrics import calculate_uncertainty



@click.command()
@click.option("--dataset", type=click.Path(exists=True), default="data/test.json")
@click.option("--out_dir", type=click.Path())
@click.option("--label", type=str)
@click.option("--structure", type=str, default=None)
@click.option("--response_dir", type=click.Path(exists=True), multiple=True)
@click.option("--scope", type=str)
def main(dataset, out_dir, response_dir, structure, label, scope):
    valid_scopes = ["label", "first_token", "response"]
    if scope not in valid_scopes:
        raise ValueError(f"Scope {scope} is invalid. Valid options: {', '.join(valid_scopes)}.")
    
    # setup out dir
    os.makedirs(out_dir, exist_ok=True)

    # load dataset
    dataset = load_json(dataset)
    dataset = organize_dataset(dataset)

    # load and process responses
    correct, wrong = {}, {}
    for fp in response_dir:
        responses = load_response_dict(fp)
        remove_invalid(responses)

        # add probabilities to responses
        calculate_probs(responses)
        find_relevant_logprobs(responses, relevant_scope=scope, structure=structure)
        calculate_uncertainty(responses)

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
    # print(list(correct.values()))
    for metric in list(correct.values())[0][0]["uncertainty"].keys():
        out_file = f"{out_dir}/{metric}_{label}.png"
        c_metric = {k : [c["uncertainty"][metric] for c in v] for k,v in correct.items()}
        w_metric = {k : [w["uncertainty"][metric] for w in v] for k,v in wrong.items()}
        


        boxplot_analysis(c_metric, w_metric, out_file)


    # do sentence confidence analysis for INCORRECT answers, format this in a nice way.
    # sentence_analysis()


if __name__=="__main__":
    main()
import json
import os
from tqdm import tqdm 
from backend.utils import load_json, load_response_dict
from backend.nli_labels import ExNLILabel
import math

import matplotlib.pyplot as plt
import numpy as np
import click
from backend.dataset_utils import organize_dataset, retrieve_annotation

def by_id(id, responses_by_seed):
    out = {}
    for k,v in responses_by_seed.items():
        out[k] = v[id]
    return out


def find_relevant_tokens(logprobs, key_phrase):
    # print("Assuming direct control")
    clean_keyphrase = "".join(key_phrase.split()) #remove whitespace to avoid issues
    token_buffer = []
    # start checking from the end of the list to reduce searching time
    for i, logprob in list(reversed(list(enumerate(logprobs)))):
        buffer_log = ""
        token = logprob["token"]
        clean_token = "".join(token.split())
        
        # check if the new subphrase can be added to buffer
        if len(token_buffer) > 0:
            # print("good", len(token_buffer))
            for j, tb in enumerate(token_buffer):
                pos, string = tb # list of token indices and the string they form
                subphrase = "".join((clean_token + string).split()) # phrase to be evaluated
                # print(subphrase, "//", clean_keyphrase)
                # print(token_buffer)

                new_pos = pos.copy()
                new_pos.insert(0, i)
                
                if clean_keyphrase in subphrase: # clean keyphrase is in the subphrase, a match is found (do "in" instead of "eq" because tokens can mass "eq" up)
                    return new_pos
                elif subphrase in clean_keyphrase: # break after this, because it alters the token buffer shape
                    
                    
                    # print(token_buffer, j, "->", end="")
                    token_buffer = token_buffer[j:] # clip the buffer to remove definite not-matches
                    
                    # update all in the rest of the buffer
                    for k, _ in enumerate(token_buffer):
                        k_pos, k_str = token_buffer[k]
                        k_pos.insert(0, i)
                        k_str = clean_token + k_str
                        token_buffer[k] = (k_pos, k_str)

                    # append newest token to buffer
                    token_buffer.append(([i], clean_token))
                    
                    # print(token_buffer)

                    break
                elif j+1 == len(token_buffer): # entire token buffer indexed without finding a match, clean token buffer with only the current token
                    # print("Next token")
                    buffer_line = f"{i} (DUMP): {clean_token} | {token_buffer[0][1]} \n"
                    buffer_log += buffer_line
                    token_buffer = [([i], clean_token)]
                    break # unnecessary but just to make it clear
        else:
            token_buffer = [([i], clean_token)]

def construct_key_phrase(prediction, structured_field=None):
    if structured_field:
        return f'{structured_field}": "{prediction}' # without starting or trailing " because tokenization is weird
    else:
        return prediction
    
def phrase_from_tokens(logprobs, token_pos):
    phrase = ""
    for tp in token_pos:
        phrase += logprobs[tp]["token"]
    return phrase

def find_relevant_logprobs(response, return_first=True, structure=None):
    # set up stuff
    
    prediction = response["prediction"]
    key_phrase = construct_key_phrase(prediction, structured_field=structure)
    # print("=== STAGE 1: SETUP")
    # print(f"Key Phrase: {key_phrase}")

    # find key phrase
    logprobs = response["logprobs"]
    token_pos = find_relevant_tokens(logprobs, key_phrase)
    # print(f"Positions: {token_pos}")
    # print(f"Phrase: {phrase_from_tokens(logprobs, token_pos)}")

    # find prediction phrase
    # print("=== STAGE 3: ISOLATE PREDICTION")
    key_phrase_tokens = logprobs[token_pos[0]:token_pos[-1]+1]
    # print([kpt["token"] for kpt in key_phrase_tokens])
    prediction_pos = find_relevant_tokens(key_phrase_tokens, prediction)
    # print(f"""Positions before correct: {prediction_pos}""")
    prediction_pos = [pp + token_pos[0] for pp in prediction_pos]
    # print(f"""Positions: {prediction_pos}
    if return_first:
        relevant_logprobs = [logprobs[pp] for pp in prediction_pos]
        return relevant_logprobs[0]
    else:
        return logprobs[prediction_pos[0]]
def logprob_to_prob(logprob, temp=1.):
    return math.exp(logprob / temp)

def get_probs(logprobs, temp=1.):
    # finds probabilities for a single token and its top logprobs
    for v in logprobs:
        # print(v)
        v["prob"] = logprob_to_prob(v["logprob"], temp=temp)
        top_logprobs = v["top_logprobs"]
        for token in top_logprobs:
            token["prob"] = logprob_to_prob(token["logprob"], temp=temp)
        # print(v["prob"])


def calculate_probs(responses, structure=None):
    for doc_id, doc_responses in tqdm(responses.items(), desc="Probabilities"):
        for hypo_id, hypo_response in doc_responses["annotation_sets"][0]["annotations"].items():
            logprobs = hypo_response["logprobs"]
            get_probs(logprobs, temp=1.)
            pred_tokens = find_relevant_logprobs(hypo_response, structure=structure)
            hypo_response["pred_tokens"] = pred_tokens


def sort_correct_wrong(dataset: dict, responses: dict):
    correct = {}
    wrong = {}

    for doc_id, doc_responses in tqdm(responses.items(), desc="Sorting"):
        for hypo_id, hypo_response in doc_responses["annotation_sets"][0]["annotations"].items():
            annotation = dataset[doc_id]["annotation_sets"][0]["annotations"][hypo_id]
            gt = annotation["choice"]
            pred = hypo_response["prediction"]
            pred_token = hypo_response["pred_tokens"]["token"]
            pred_prob = hypo_response["pred_tokens"]["prob"]
            
            if gt == pred:
                target_dict = correct
            else:
                target_dict = wrong

            if pred_token not in target_dict.keys():
                target_dict[pred] = []
            target_dict[pred].append(pred_prob)
                
    print(f"Num correct: {sum(len(v) for v in correct.values())}")
    print(f"Num wrong:   {sum(len(v) for v in wrong.values())}")

    return correct, wrong


def boxplot_analysis(correct: dict, wrong: dict, save_file: str, ylabel: str ="Probability"):

    d = list(correct.values()) + list(wrong.values())
    labels = [
        f"Correct {k}" for k in correct.keys()
        ] + [
        f"Wrong {k}" for k in wrong.keys()
        ]
    print(len(d), len(labels))
    
        

    fig = plt.figure(figsize=(10,7)) 
    plt.boxplot(d, labels=labels)
        
    # ax.set_xticklabels(labels)
    # ax.set_xlabel("Results")
    # ax.set_ylabel(ylabel)

    fig.savefig(save_file)

def sentence_analysis(dataset, responses):
    pass

def cross_model_judgement(dataset, responses):
    pass

def remove_invalid(responses: dict):
    # remove all invalid answers, as they are not going to be useful for logprobs.
    counter = 0
    for doc_id, doc_responses in responses.items():
        annotations = doc_responses["annotation_sets"][0]["annotations"]
        hypotheses = list(annotations.keys())
        for hypo_id in hypotheses:
            if annotations[hypo_id]["prediction"] == ExNLILabel.INVALID_ANSWER.to_anno_name():
                del annotations[hypo_id]
                counter += 1

    print(f"Removed {counter} invalid answers!")

@click.command()
@click.option("--dataset", type=click.Path(exists=True), default="data/test.json")
@click.option("--out_file", type=click.Path())
@click.option("--structure", type=str)
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
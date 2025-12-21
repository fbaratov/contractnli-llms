"""
All metrics are implemented based on equations collected by Shorinwa et al. 2025
"""

import numpy as np
import pprint

from tqdm import tqdm

def average_logprob(logprobs: np.array) -> np.double:
    return -np.average(logprobs)

def perplexity(logprobs: np.array) -> np.double:
    return np.exp(average_logprob(logprobs))

def maximum_logprob(logprobs: np.array) -> np.double:
    return np.max(-logprobs)

def response_improbability(logprobs: np.array) -> np.double:
    return 1 - np.prod(logprobs)

def probability_distribution(top_logprobs: np.array) -> np.double:
    top_probs = np.exp(top_logprobs)
    product = top_probs * top_logprobs
    return -np.sum(product)

def entropy(top_logprobs_per_token: np.array) -> np.double:
    prob_dist = np.apply_along_axis(probability_distribution, 1, top_logprobs_per_token)
    return np.max(prob_dist)

def all_metrics(top_logprobs_per_token: np.array) -> np.double:
    
    uncertainty_dict = {}
    logprobs = top_logprobs_per_token[:,0]

    for fn in [entropy]: # functions requiring multiple top logprobs per token
        uncertainty_dict[fn.__name__] = fn(top_logprobs_per_token)
    
    # functions requiring just the top logprob
    for fn in [average_logprob, perplexity, maximum_logprob, response_improbability]:
        uncertainty_dict[fn.__name__] = fn(logprobs)

    return uncertainty_dict

def metrics_from_list(phrase_tokens: list, n=None) -> dict:
    logprobs_list = []
    n = min([len(pt["top_logprobs"]) for pt in phrase_tokens]) if n is None else n
    for pt in phrase_tokens:
        pt_logprobs = []
        pt_logprobs.append(pt["logprob"])
        for token in pt["top_logprobs"][:n+1]:
            pt_logprobs.append(token["logprob"])
        logprobs_list.append(pt_logprobs)
    
    top_logprobs_per_token = np.array(logprobs_list)
    # print(top_logprobs_per_token)
    return all_metrics(top_logprobs_per_token)

def calculate_uncertainty(responses):
    for doc_id, doc_responses in tqdm(responses.items(), desc="Probabilities"):
        for hypo_id, hypo_response in doc_responses["annotation_sets"][0]["annotations"].items():
            pred_tokens = hypo_response["pred_tokens"]
            metrics = metrics_from_list(pred_tokens)
            hypo_response["uncertainty"] = metrics

if __name__=="__main__":
    top_logprobs_per_token = np.array([
        [-0.1,-0.2,-0.3,-0.4,-0.5],
        [-0.1,-0.2,-0.3,-0.4,-0.5],
        [-0.1,-0.2,-0.3,-0.4,-0.5],
        [-0.1,-0.2,-0.3,-0.4,-0.5],
        [-0.1,-0.2,-0.3,-0.4,-0.5],
        [-0.1,-0.2,-0.3,-0.4,-0.5]
    ])
    
    pprint.pprint(all_metrics(top_logprobs_per_token))

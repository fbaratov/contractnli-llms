import traceback

from tqdm import tqdm 
from backend.nli_labels import ExNLILabel
import math

def find_token_sublists(tokens, target, clean_whitespace=True, overlapping=False):
    """
    Given a list of tokens (strings, or dicts with a "token" key) and a target
    string, find all contiguous sublists of tokens whose concatenation
    contains `target`.

    Returns a list of matches, each a list of token indices (contiguous),
    in the order they occur. Empty list if no match is found.
    """
    def get_text(tok):
        return tok["token"] if isinstance(tok, dict) else tok

    clean = (lambda s: "".join(s.split())) if clean_whitespace else (lambda s: s)

    target_clean = clean(target)
    if not target_clean:
        return []

    # Build the full concatenated string, and a parallel array mapping
    # each character position back to the token index it came from.
    full_text_parts = []
    char_to_token = []
    for idx, tok in enumerate(tokens):
        t = clean(get_text(tok))
        full_text_parts.append(t)
        char_to_token.extend([idx] * len(t))
    full_text = "".join(full_text_parts)

    matches = []
    start = 0
    while True:
        pos = full_text.find(target_clean, start)
        if pos == -1:
            break
        end = pos + len(target_clean) - 1
        token_start = char_to_token[pos]
        token_end = char_to_token[end]
        matches.append(list(range(token_start, token_end + 1)))
        start = pos + 1 if overlapping else pos + len(target_clean)

    return matches

def get_token_sublist(tokens, target, clean_whitespace=True):
    """Convenience wrapper: returns the first match's token indices, or None."""
    matches = find_token_sublists(tokens, target, clean_whitespace=clean_whitespace)
    return matches[-1] if matches else None

# def get_token_sublist_objects(tokens, target, clean_whitespace=True):
#     """Like find_first_token_sublist, but returns the token dicts/strings themselves."""
#     indices = find_first_token_sublist(tokens, target, clean_whitespace=clean_whitespace)
#     if indices is None:
#         return None
#     return [tokens[i] for i in indices]

def find_relevant_tokens(logprobs, key_phrase):
    match = get_token_sublist(logprobs, key_phrase)
    if match is None:
        raise ValueError(f"key_phrase {key_phrase!r} not found in logprobs")
    return match

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

def find_relevant_logprobs_per_response(response, return_first=False, structure=None):
    # set up stuff
    prediction = response["prediction"]
    key_phrase = construct_key_phrase(prediction, structured_field=structure)
    logprobs = response["logprobs"]
    key_phrase_tokens = None
    # try:
    #     token_pos = find_relevant_tokens(logprobs, key_phrase)
    # except Exception as e:
    #     print(f"""
    #     EXCEPTION WHEN GETTING TOKEN_POS
    #     VALUES
    #         {key_phrase=}
    #     """)
    #     raise e
    # # print(token_pos)
    # key_phrase_tokens = logprobs[token_pos[0]:token_pos[-1]+1]
    try:
        prediction_pos = find_relevant_tokens(logprobs, prediction)
    except Exception as e:
        print(f"""
        EXCEPTION WHEN GETTING PREDICTION_POS
        VALUES
            {key_phrase=},
            {key_phrase_tokens=},
            {prediction=}
        """)
        raise e

    # prediction_pos = [pp + token_pos[0] for pp in prediction_pos]
    # print(prediction_pos)
    # return the right thing
    if return_first:
        relevant_pos = prediction_pos[0]
        return [logprobs[relevant_pos]]
    else:
        relevant_logprobs = [logprobs[pp] for pp in prediction_pos]
        # print([lp["token"] for lp in relevant_logprobs])
        return relevant_logprobs

def find_relevant_logprobs(responses, structure=None, relevant_scope=None):
    for doc_id, doc_responses in tqdm(responses.items(), desc="Probabilities"):
        for hypo_id, hypo_response in doc_responses["annotation_sets"][0]["annotations"].items():
            if hypo_response["prediction"] == ExNLILabel.INVALID_ANSWER.to_anno_name():
                del responses[doc_id]["annotation_sets"][0]["annotations"][hypo_id]
            if relevant_scope in ["first_token", "label"]:
                try:
                    hypo_response["pred_tokens"] = find_relevant_logprobs_per_response(hypo_response, structure=structure, return_first = (relevant_scope=="first_token"))
                except TypeError as e:
                    print("Doc and hypo:", doc_id, hypo_id)
                    print(traceback.format_exc())
                    raise TypeError()
            else:
                hypo_response["pred_tokens"] = hypo_response["logprobs"]

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


def calculate_probs(responses):
    for doc_id, doc_responses in tqdm(responses.items(), desc="Probabilities"):
        for hypo_id, hypo_response in doc_responses["annotation_sets"][0]["annotations"].items():
            logprobs = hypo_response["logprobs"]
            get_probs(logprobs, temp=1.)
            

def sort_correct_wrong(dataset: dict, responses: dict):
    correct = {}
    wrong = {}

    for doc_id, doc_responses in tqdm(responses.items(), desc="Sorting"):
        for hypo_id, hypo_response in doc_responses["annotation_sets"][0]["annotations"].items():

            # extarct relevant values
            annotation = dataset[doc_id]["annotation_sets"][0]["annotations"][hypo_id]
            gt = annotation["choice"]
            pred = hypo_response["prediction"]
            # determine if correct or wrong prediction
            if gt == pred:
                target_dict = correct
            else:
                target_dict = wrong

            # sort based on prediction
            if pred not in target_dict.keys():
                target_dict[pred] = []
            target_dict[pred].append(hypo_response) # append annotation to get all info
                
    print(f"Num correct: {sum(len(v) for v in correct.values())}")
    print(f"Num wrong:   {sum(len(v) for v in wrong.values())}")

    return correct, wrong

def remove_invalid(responses: dict):
    # remove all invalid answers, as they are not going to be useful for logprobs.
    counter = 0
    for doc_id, doc_responses in responses.items():
        if len(doc_responses["annotation_sets"]) > 1:
            print(doc_id, len(doc_responses["annotation_sets"]))

    for doc_id, doc_responses in responses.items():
        annotations = doc_responses["annotation_sets"][0]["annotations"]
        hypotheses = list(annotations.keys())
        for hypo_id in hypotheses:
            if annotations[hypo_id]["prediction"] in (ExNLILabel.INVALID_ANSWER.to_anno_name(), "InvalidAnswer") or annotations[hypo_id]["response"] == "":
                print(doc_id, hypo_id)
                del responses[doc_id]["annotation_sets"][0]["annotations"][hypo_id]
                counter += 1

    print(f"Removed {counter} invalid answers!")

    return responses
from tqdm import tqdm 
from backend.nli_labels import ExNLILabel
import math

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
                    print("MATCH")
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
                    
                    print(token_buffer)

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

def find_relevant_logprobs_per_response(response, return_first=False, structure=None):
    # set up stuff
    prediction = response["prediction"]
    key_phrase = construct_key_phrase(prediction, structured_field=structure)
    logprobs = response["logprobs"]
    token_pos = find_relevant_tokens(logprobs, key_phrase)
    key_phrase_tokens = logprobs[token_pos[0]:token_pos[-1]+1]
    prediction_pos = find_relevant_tokens(key_phrase_tokens, prediction)
    prediction_pos = [pp + token_pos[0] for pp in prediction_pos]
    
    # return the right thing
    if return_first:
        relevant_pos = prediction_pos[0]
        return [logprobs[relevant_pos]]
    else:
        relevant_logprobs = [logprobs[pp] for pp in prediction_pos]
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
                    print(e)
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
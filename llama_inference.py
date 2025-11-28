# load quantized model via llama cpp if the original don't fit
# take given input and process it via tokenization/utf-8 encoding if necessary
# from here 
import os
import sys

# Load model directly
import torch
from llama_cpp import Llama
import time



##### UTILS ####

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()




def load_model(model):
    repo_id, filename = model["repo_id"], model["filename"]
    print("Loading model....")
    with suppress_stdout_stderr():
        start_time = time.time()
        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            logits_all=True
            )
        end_time = time.time()
        tok = llm.tokenizer()
    print(f"Model loaded in {round(end_time - start_time, 3)} seconds")


def inference(prompt, llm, tok, allowed_tokens=[' True', ' False'], max_tokens=1, num_logprobs=1000000):
    inputs = tok.encode(prompt, special=False)

    with torch.no_grad():
        with suppress_stdout_stderr():
            out = llm(inputs, max_tokens=max_tokens, logprobs=num_logprobs)
            logprobs = out["choices"][0]["logprobs"]["top_logprobs"][0]

    logits = []
    for token in allowed_tokens:
        logits.append(logprobs[token])
    logits = torch.tensor(logits)
    logprobs = torch.softmax(logits, dim=-1)

    return dict(zip(allowed_tokens, logprobs))

    # for token, logprob in zip(allowed_tokens, logprobs):
    #     print(f"{token}: {logprob}")

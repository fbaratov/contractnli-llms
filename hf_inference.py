from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# load model
# Load model directly
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(config, device=device):
    quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent overflow in gradients
                        ) if config["quantize"] is True else None
    repo_id = config["model"]
    print("Loading model....")
    start_time = time.time()
    llm = AutoModelForImageTextToText.from_pretrained(repo_id, quantization_config=quantization_config).to(device)
    tok = AutoProcessor.from_pretrained(repo_id)
    end_time = time.time()
    print(f"Model loaded in {round(end_time - start_time, 3)} seconds")
    return llm, tok
 

# perform inference and retrieve logits 

def inference(prompt, model, tok, config, device=device):
    
    model.eval()
    
    # inputs = tok(prompt, return_tensors="pt").to(device)
    print(prompt)
    with torch.no_grad():
        transformers.set_seed(config["seed"])
        out = model(**prompt, max_new_tokens=1)
    logits = out.logits          # [batch, seq_len, vocab_size]
    next_token_logits = logits[:, -1, :]  # last position
    print(out)
    print(next_token_logits)
    ### assemble allowed id's
        
    label2tokens = {
        "NEG": tok.encode("no", add_special_tokens=False),
        "POS": tok.encode("yes", add_special_tokens=False),
    }

    allowed_ids = list({tid for tids in label2tokens.values() for tid in tids})


    ### label from logits

    allowed = torch.tensor(allowed_ids)
    label_logits = next_token_logits[:, allowed]
    # [batch, n_labels]
    probs = label_logits.softmax(dim=-1)

    
    return dict(zip(allowed, probs))

if __name__=="__main__":
    config = {
        "model": "google/gemma-3-27b-it",
        "quantize": True,
        "seed": 0
    }
    llm, tok = load_model(config)

    # prompt = tok.apply_chat_template([
    #     {"role": "user", "content": "Does France exist?"}
    #     ],
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt"
    # )
    # results = inference(
    #     prompt=prompt,
    #     model=llm,
    #     tok=tok,
    #     config=config,
    #     device=device
    # )

    # print(results)
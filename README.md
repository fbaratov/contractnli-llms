# Installation (WIP)

1. Install the latest version of Ollama.
2. Install any additional package dependencies that arise. 
3. Download the ContractNLI dataset and place into a `data' folder in the root directory of the repository.

# Running

* main.py - run to do inference with a given config file (see configs folder for examples). run/batch.sh to run a batch of configs and output them all neatly into a given output directory (has to be given as an argument, like for main.py)

* evaluate_responses.py - run to evaluate a given set of responses. run/evaluate_all.sh to do this for all sets of responses in a given directory.

* avg_eval.py - averages results for a given set of response directories. use run/avg_eval.sh to make this easier and do it neatly for a given directory (will always average across seeds for a given model config)

* notebooks/uq_boxplots.ipynb - contains code/outputs for UQ boxplots. Not super clean, but runs.
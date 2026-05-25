# Introduction
This repository is used for quickly running inference on the ContractNLI and NLI4Wills datasets using an Ollama backend, as well as for evaluation and analysis of results.

# Installation (WIP)

1. Install the latest version of Ollama.
2. Install any additional package dependencies that arise. 
3. Download the ContractNLI dataset and place into a `data' folder in the root directory of the repository.

This installation process is fairly reliable, and has been used to successfully set this repository up on several Linux and Windows systems.

# Running

## Inference

Use the following command to run inference with a given model config (see ```configs``` for some options). 

```
python main.py --model_config PATH_TO_CONFIG --output_dir OUTPUT_DIR
```

Model options can be provided via the config file, and additional arguments can also be provided for ```main.py```, including seed, temperature, and the ollama server address (see that script for more details).


## Evaluation

Evaluation can be ran using ```evaluate_responses.py``` or ```compute_nli4wills_metrics.py```, depending on the dataset that you are using.

For ContractNLI:

```
python evaluate_responses.py --response_dir RESPONSE_DIR --eval_dir EVAL_DIR --dataset DATASET
```
* ```RESPONSE_DIR```: path to directory containing response jsons,
* ```EVAL_DIR```: output directory,
* ```DATASET```: path to ContractNLI dataset file. 

For NLI4Wills-Idaho and NLI4Wills-Tennessee:
```
python compute_nli4wills_metrics.py --response_dir RESPONSE_DIR --eval_dir EVAL_DIR --dataset DATASET
```
* ```RESPONSE_DIR```: path to directory containing response jsons,
* ```EVAL_DIR```: output directory,
* ```DATASET```: path to json dataset file. 

### Getting average evaluations:

Use ```avg_eval.py``` for ContractNLI and ```compute_nli4wills_average.py``` for the NLI4Wills datasets. The arguments are the same for both:

```
python avg_eval.py -m metrics1.json -m metrics2.json -m metrics3.json -o out.json
```
Use ```-m``` to specify metrics files to average, and ```-o``` to specify the output file.

# Notebooks


* ```notebooks/logprob_analysis.ipynb``` - used for running Mann-Whitney U Tests and formatting outputs as a LaTeX table.
* ```notebooks/temp_plots.ipynb``` - used for compiling files that are used for LaTeX temperature plots.
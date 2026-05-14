#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=alice/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 # 24 hours runtime just in case
#SBATCH --partition=gpu-l4-24g # it will run on anything but this is more likely to work right


# # SBATCH --partition=gpu-mig-40g # it will run on anything but this is more likely to work right





# get conda right
conda init
source ~/.bashrc

conda deactivate
conda activate ollama


# run ollama server (use complicated ALICE tutorial version to make sure it gets done correctly)
# check that the port by ollama is not already in use
function test_ollama_port_closed {
  for i in {11434..11454} # will test start port number 11434 till end port number 11454 if ollama is already running on that port
  do 
    TEST_OLLAMA_PORT=$(nc -z 127.0.0.1 "$i" && echo open || echo closed) #this returns "open" if the port is open and "closed if the port is closed
    if [ "$TEST_OLLAMA_PORT" != "open" ] 
    then
      echo $i
      break
    fi
  done
}
 
#Run Llama

# define alias because why not
ollama="/home/baratovfs/ollama/bin/ollama"
 
OLLAMA_PORT=$(test_ollama_port_closed) #find the first open port from the test_ollama_port function
echo "Running ollama as: OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT} #ollama serve &"
OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT} $ollama serve &

sleep 30 # Adjust this value if needed

# run provided command with server added
srun ${@} --server 127.0.0.1:${OLLAMA_PORT}

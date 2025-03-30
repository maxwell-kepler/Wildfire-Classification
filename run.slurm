#!/bin/bash
#SBATCH --job-name=Wildfires
#SBATCH --output=%j.out              
#SBATCH --error=%j.err               
#SBATCH --partition=gpu              
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=2           
#SBATCH --gres=gpu:1                
#SBATCH --mem=16G                   
#SBATCH --time=20:00:00             
#SBATCH --qos=gpulimit             

# Source conda initialization
source /home/maxwell.kepler/miniforge3/bin/activate

# Activate your environment
conda activate pytorch_env

# Navigate to working directory
cd /home/maxwell.kepler/TransferLearning/FinalProject

# Run script
python CNN.py
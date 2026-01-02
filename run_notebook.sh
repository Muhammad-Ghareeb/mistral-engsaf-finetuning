#!/bin/sh
#SBATCH --job-name=mistral-training
#SBATCH --account=ejust005u1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=mistral_engsaf_finetuning.out
#SBATCH --error=mistral_engsaf_finetuning.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load Anaconda3 module
module load Anaconda3

# Activate the BA-HPC conda environment
source activate /share/apps/conda_envs/ba-hpc

# Navigate to your working directory
cd /cluster/users/ejust005u1/QA-Feedback

# Verify GPU is available
echo "Checking GPU availability..."
nvidia-smi

# Print Python version
echo "Python version: $(python --version)"

# Execute the notebook automatically
echo "Starting notebook execution..."
jupyter nbconvert --to notebook --execute mistral_engsaf_finetuning.ipynb --output mistral_engsaf_finetuning_executed.ipynb

# Alternative: Execute and convert to HTML (easier to view results)
# jupyter nbconvert --to html --execute mistral_engsaf_finetuning.ipynb --output mistral_engsaf_finetuning_results.html

echo "Notebook execution completed at: $(date)"
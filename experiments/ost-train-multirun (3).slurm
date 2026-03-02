#!/bin/bash
#SBATCH --job-name=ost_PH_multirun
#SBATCH --partition=normal                # the DGX only belongs in the 'gpu' partition
#SBATCH --output=/scratch/spothago/pfiles/ost_multirun_output.%j.txt
#SBATCH --error=/scratch/spothago/pfiles/ost_multirun_error.%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                     # up to 128; 
#SBATCH --mem-per-cpu=20GB                     # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=120:00:00                       # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL              # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=spothago@gmu.edu            # Put your GMU email address here

set -x
umask 0027


module load gnu10
module load python
module load cudnn

source /scratch/spothago/wandb-agent/bin/activate

# Get the Sweep ID from the argument passed by the master script
SWEEP_ID=$1

# 3. Run the agent (count=1 means it runs 1 experiment and then the job ends)
# If you want one job to run multiple experiments until the time limit, remove --count 1 
wandb agent Ost_new_lower/$SWEEP_ID --count 1000
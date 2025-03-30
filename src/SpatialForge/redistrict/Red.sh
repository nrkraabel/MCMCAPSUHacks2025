#!/bin/bash
#SBATCH --job-name=cohesion-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=240GB
#SBATCH --time=8:00:00
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --output=cohesion-test-%j.out

# Set up working directory
WORK_DIR="/storage/home/nrk5343/work/MCMCAPSUHacks2025"
cd $WORK_DIR

# Create output directory
mkdir -p cohesion_test_results

# Run the test
python run_cohesion.py --config cohesion.json --output-dir ../cohesion_test_results --scenario test_cohesive

echo "Test completed!"
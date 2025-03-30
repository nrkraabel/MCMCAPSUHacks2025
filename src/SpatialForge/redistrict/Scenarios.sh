#!/bin/bash
#SBATCH --job-name=gerrymander-scenarios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=240GB
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --output=gerrymander-scenarios-%j.out

# Print some debug information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Available CPU cores: $(nproc)"
echo "Memory: $(free -h)"

# Load necessary modules (adjust as needed for your environment)
module purge
module load ffmpeg  # For creating animations

# Set up working directory
WORK_DIR="/storage/home/nrk5343/work/MCMCAPSUHacks2025"
cd $WORK_DIR

# Ensure output directories exist
RESULTS_DIR="$WORK_DIR/results"
mkdir -p $RESULTS_DIR

# Activate virtual environment if needed
source .venv/bin/activate

# 1. Fair Redistricting
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=fair-map
#SBATCH --cpus-per-task=40
#SBATCH --mem=240GB
#SBATCH --time=24:00:00
#SBATCH --output=fair-map-%j.out
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri

cd $WORK_DIR
source .venv/bin/activate
python /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/run_batch_scenarios.py --config /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/scanarios_config2.json --output-dir $RESULTS_DIR --scenario fair_redistricting
EOT

# 2. Republican Gerrymander
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=red-map
#SBATCH --cpus-per-task=40
#SBATCH --mem=240GB
#SBATCH --time=24:00:00
#SBATCH --output=red-map-%j.out
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri

cd $WORK_DIR
source .venv/bin/activate
python /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/run_batch_scenarios.py --config /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/scanarios_config2.json --output-dir $RESULTS_DIR --scenario red_gerrymander
EOT

# 3. Democratic Gerrymander
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=blue-map
#SBATCH --cpus-per-task=40
#SBATCH --mem=240GB
#SBATCH --time=24:00:00
#SBATCH --output=blue-map-%j.out
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri

cd $WORK_DIR
source .venv/bin/activate
python /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/run_batch_scenarios.py --config /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/scanarios_config2.json --output-dir $RESULTS_DIR --scenario blue_gerrymander
EOT

# 4. Incumbent Protection
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=incumbent-map
#SBATCH --cpus-per-task=40
#SBATCH --mem=240GB
#SBATCH --time=24:00:00
#SBATCH --output=incumbent-map-%j.out
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri

cd $WORK_DIR
source .venv/bin/activate
python /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/run_batch_scenarios.py --config /storage/home/nrk5343/work/MCMCAPSUHacks2025/Redistricting/scanarios_config2.json --output-dir $RESULTS_DIR --scenario incumbent_protection
EOT

# Optional: Run a parallel version that processes all scenarios at once
# This might be useful for testing/debugging but will use multiple cores per scenario
# which might not be optimal
#python run_batch_scenarios.py --config scenarios_config.json --output-dir $RESULTS_DIR --parallel

echo "All scenario jobs submitted at $(date)"
echo "Check individual job logs for progress"
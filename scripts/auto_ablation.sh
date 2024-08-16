#!/bin/bash

# Define the .screenrc file path
SCREENRC="$HOME/.screenrc"
# Check if .screenrc exists and contains the necessary configuration
if ! grep -q "^mousetrack on" "$SCREENRC"; then
    echo "Adding mouse support configuration to $SCREENRC"
    echo "mousetrack on" >> "$SCREENRC"
else
    echo "Mouse support is already configured in $SCREENRC"
fi

# Create a directory for log files if it does not exist
mkdir -p screen_log

# RoboDuet
screen -dmS RoboDuet_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 2423 --run_name RoboDuet --sim_device cuda:0 > screen_log/RoboDuet_Seed2423.log; exec bash"
screen -dmS RoboDuet_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 8765 --run_name RoboDuet --sim_device cuda:1 > screen_log/RoboDuet_Seed8765.log; exec bash"
screen -dmS RoboDuet_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 3779 --run_name RoboDuet --sim_device cuda:2 > screen_log/RoboDuet_Seed3779.log; exec bash"

# Cooperated
screen -dmS Cooperated_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 2423 --run_name Cooperated --sim_device cuda:0 > screen_log/Cooperated_Seed2423.log; exec bash"
screen -dmS Cooperated_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Cooperated --sim_device cuda:1 > screen_log/Cooperated_Seed8765.log; exec bash"
screen -dmS Cooperated_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 3779 --run_name Cooperated --sim_device cuda:2 > screen_log/Cooperated_Seed3779.log; exec bash"

# TwoStage
screen -dmS TwoStage_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/unified_train.py --num_envs 4096 --seed 2423 --run_name TwoStage --sim_device cuda:3 > screen_log/TwoStage_Seed2423.log; exec bash"
screen -dmS TwoStage_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --num_envs 4096 --seed 8765 --run_name TwoStage --sim_device cuda:4 > screen_log/TwoStage_Seed8765.log; exec bash"
screen -dmS TwoStage_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --num_envs 4096 --seed 3779 --run_name TwoStage --sim_device cuda:5 > screen_log/TwoStage_Seed3779.log; exec bash"

# Baseline
screen -dmS Baseline_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 2423 --run_name Baseline --sim_device cuda:0 > screen_log/Baseline_Seed2423.log; exec bash"
screen -dmS Baseline_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Baseline --sim_device cuda:0 > screen_log/Baseline_Seed8765.log; exec bash"
screen -dmS Baseline_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 3779 --run_name Baseline --sim_device cuda:5 > screen_log/Baseline_Seed3779.log; exec bash"
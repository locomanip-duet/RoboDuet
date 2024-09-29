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
# screen -dmS RoboDuet_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 2423 --run_name RoboDuet --sim_device cuda:0; exec bash"
screen -dmS RoboDuet_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 8765 --run_name RoboDuet_guide --sim_device cuda:0 --headless --guide; exec bash"
screen -dmS RoboDuet_Seed8765_2 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 8765 --run_name RoboDuet_noguide --sim_device cuda:1 --headless; exec bash"

# screen -dmS RoboDuet_Seed5078 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 5078 --run_name RoboDuet --sim_device cuda:2; exec bash"

# # Cooperated
# screen -dmS Cooperated_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 2423 --run_name Cooperated --sim_device cuda:0; exec bash"
screen -dmS Cooperated_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Cooperated_guide --sim_device cuda:2 --headless --guide; exec bash"
screen -dmS Cooperated_Seed8765_2 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Cooperated_noguide --sim_device cuda:3 --headless; exec bash"
# screen -dmS Cooperated_Seed5078 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 5078 --run_name Cooperated --sim_device cuda:1; exec bash"

# # TwoStage
# screen -dmS TwoStage_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --num_envs 4096 --seed 2423 --run_name TwoStage --sim_device cuda:0; exec bash"
# screen -dmS TwoStage_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --num_envs 4096 --seed 8765 --run_name TwoStage --sim_device cuda:1; exec bash"
# screen -dmS TwoStage_Seed5078 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --num_envs 4096 --seed 5078 --run_name TwoStage --sim_device cuda:2; exec bash"

# # Baseline
# screen -dmS Baseline_Seed2423 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 2423 --run_name Baseline --sim_device cuda:3; exec bash"
# screen -dmS Baseline_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Baseline --sim_device cuda:4; exec bash"
# screen -dmS Baseline_Seed5078 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 5078 --run_name Baseline --sim_device cuda:5; exec bash"

#!/bin/bash

# Activate conda environment in each screen session and run the commands, capturing output to log files
mkdir screen_log
# RoboDuet
#screen -dmS RoboDuet_Seed2 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 2 --run_name RoboDuet --sim_device cuda:0 > screen_log/RoboDuet_Seed2.log"
#screen -dmS RoboDuet_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 8765 --run_name RoboDuet --sim_device cuda:1 > screen_log/RoboDuet_Seed8765.log"
#screen -dmS RoboDuet_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/auto_train.py --num_envs 4096 --seed 3779 --run_name RoboDuet --sim_device cuda:2 > screen_log/RoboDuet_Seed3779.log"

# Cooperated
#screen -dmS Cooperated_Seed2 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 2 --run_name Cooperated --sim_device cuda:0 > screen_log/Cooperated_Seed2.log"
#screen -dmS Cooperated_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Cooperated --sim_device cuda:1 > screen_log/Cooperated_Seed8765.log"
#screen -dmS Cooperated_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 3779 --run_name Cooperated --sim_device cuda:2 > screen_log/Cooperated_Seed3779.log"

# TwoStage
# screen -dmS TwoStage_Seed2 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate legdog && python scripts/unified_train.py --num_envs 4096 --seed 2 --run_name TwoStage --sim_device cuda:3 > screen_log/TwoStage_Seed2.log"
#screen -dmS TwoStage_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --num_envs 4096 --seed 8765 --run_name TwoStage --sim_device cuda:4 > screen_log/TwoStage_Seed8765.log"
#screen -dmS TwoStage_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --num_envs 4096 --seed 3779 --run_name TwoStage --sim_device cuda:5 > screen_log/TwoStage_Seed3779.log"

# Baseline
screen -dmS Baseline_Seed2 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 2 --run_name Baseline --sim_device cuda:0 > screen_log/Baseline_Seed2.log"
#screen -dmS Baseline_Seed8765 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Baseline --sim_device cuda:4 > screen_log/Baseline_Seed8765.log"
#screen -dmS Baseline_Seed3779 bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacgym && python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 3779 --run_name Baseline --sim_device cuda:5 > screen_log/Baseline_Seed3779.log"
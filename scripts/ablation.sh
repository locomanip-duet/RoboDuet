python scripts/auto_train.py --num_envs 4096 --seed 2 --run_name RoboDuet --sim_device cuda:0
python scripts/auto_train.py --num_envs 4096 --seed 8765 --run_name RoboDuet --sim_device cuda:1
python scripts/auto_train.py --num_envs 4096 --seed 3779 --run_name RoboDuet --sim_device cuda:2

python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 2 --run_name Cooperated --sim_device cuda:3
python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Cooperated --sim_device cuda:4
python scripts/auto_train.py --wo_two_stage --num_envs 4096 --seed 3779 --run_name Cooperated --sim_device cuda:5

python scripts/unified_train.py --num_envs 4096 --seed 2 --run_name TwoStage --sim_device cuda:3
python scripts/unified_train.py --num_envs 4096 --seed 8765 --run_name TwoStage --sim_device cuda:4
python scripts/unified_train.py --num_envs 4096 --seed 3779 --run_name TwoStage --sim_device cuda:5

python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 2 --run_name Baseline --sim_device cuda:3
python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 8765 --run_name Baseline --sim_device cuda:4
python scripts/unified_train.py --wo_two_stage --num_envs 4096 --seed 3779 --run_name Baseline --sim_device cuda:5

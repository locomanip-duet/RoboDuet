# ROBODUET

1. 首先要修改 auto_ablation.sh 中的 `conda activate isaacgym` 的环境名字，同时修改 cuda id
2. 估计是要修改 auto_train.py 和 unified_train.py 的 wandb.init 的 entity

3. 选完 auto_ablation.sh 中要跑的几个后
```
bash scripts/auto_ablation.sh
```
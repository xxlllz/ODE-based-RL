# ODE-based Recurrent Model-free Reinforcement Learning for POMDPs
# 3D Face Reconstruction with the Geometric Guidance of Facial Part Segmentation


The code is highly built on [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://github.com/twni2016/pomdp-baselines)
The code has not been fully refactored.

## New
Maybe there are some bugs or issues that have not been solved, I will continue to work on these issues.

### "Standard" POMDP

{Ant,Cheetah,Hopper,Walker}-{P,V} in the paper, corresponding to `configs/pomdp/<ant|cheetah|hopper|walker>_blt/<p|v>`, which requires PyBullet. We also provide Pendulum environments for sanity check.

Take Ant-P as an example:
```bash
# Run our implementation
python main.py --cfg configs/pomdp/ant_blt/p/rnn.yml --algo sac
# Oracle: we directly use Table 1 results (SAC w/ unstructured row) in https://arxiv.org/abs/2005.05719 as it is well-tuned
``` 

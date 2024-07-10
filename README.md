# ODE-based-RL
[ODE-based Recurrent Model-free Reinforcement Learning for POMDPs](https://arxiv.org/abs/2309.14078) (Neurips 2023)


The code is built on [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://github.com/twni2016/pomdp-baselines).


### News
Maybe there are some bugs or issues that have not been solved, I will continue to work on these problems.

### "Standard" POMDP

{Ant,Cheetah,Hopper,Walker}-{P,V} in the paper, corresponding to `configs/pomdp/<ant|cheetah|hopper|walker>_blt/<p|v>`, which requires PyBullet. We also provide Pendulum environments for sanity check.

Take Ant-P as an example:
```bash
# Run our implementation
python main.py --cfg configs/pomdp/ant_blt/p/rnn.yml --algo sac
# Oracle: we directly use Table 1 results (SAC w/ unstructured row) in https://arxiv.org/abs/2005.05719 as it is well-tuned
```


## Reference  
We highly appreciate your act of staring and citing. Your attention to detail and recognition is greatly valued.  
  
```bibtex 
@article{zhao2023ode,
  title={Ode-based recurrent model-free reinforcement learning for pomdps},
  author={Zhao, Xuanle and Zhang, Duzhen and Liyuan, Han and Zhang, Tielin and Xu, Bo},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={65801--65817},
  year={2023}
}
```

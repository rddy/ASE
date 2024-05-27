## ASE - Assistive State Estimation

[ASE](https://arxiv.org/abs/2008.02840) is an algorithm for replacing uninformative observations
with synthetic observations that are optimized to induce accurate beliefs about the current state
in the user. This codebase implements ASE in four domains:

1. An MNIST classification task
2. A Car Racing video game
3. A Lunar Lander video game
4. 2D navigation in [Habitat](https://aihabitat.org/) environments

## Usage

1.  Clone `sensei` into your home directory `~`
2.  Download [data.zip](https://drive.google.com/file/d/1FNFYJvZN2ioloBqsFgBFHAkOsAzC0egL/view?usp=sharing) and decompress it into `sensei/`
3.  Download [labeling-data.zip](https://drive.google.com/file/d/1NEPBE1aVey6awxkSDHBAV1D5de-xZ7N7/view?usp=sharing) and decompress it into `sensei/labeling/`
3.  Setup an Anaconda virtual environment with `conda create -n senseienv python=3.6`
4.  Install dependencies with `pip install -r requirements.txt` and `cp deps/box2d/*.py your_conda_install_dir/envs/senseienv/lib/python3.6/site-packages/gym/envs/box2d/; cp deps/rendering.py your_conda_install_dir/envs/senseienv/lib/python3.6/site-packages/gym/envs/classic_control/`
5.  Install [Habitat](https://github.com/facebookresearch/habitat-api)
6.  Install the `sensei` package with `python setup.py install`
7.  Jupyter notebooks in `sensei/notebooks` provide an entry-point to the code base, where you can
    play around with the environments and reproduce the figures from the paper.
8.  To run the user study script, use `python scripts/run_user_study.py 13 [gridworld|car|lander]`
9.  To run the MNIST user study, run `python labeling/app.py`, then open [http://localhost:5000/?userid=13](http://localhost:5000/?userid=13)

## Citation

If you find this software useful in your work, we kindly request that you cite the following
[paper](https://arxiv.org/abs/2008.02840):

```
@article{ase2020,
  title={Assisted Perception: Optimizing Observations to Communicate State},
  author={Reddy, Siddharth and Levine, Sergey and Dragan, Anca D.},
  journal={arXiv preprint arXiv:2008.02840},
  year={2020}
}
```

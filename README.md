# AXIOM

This repository contains the code to reproduce the paper AXIOM: Learning to Play Games in Minutes with
Expanding Object-Centric Models.


## Installation

Install using pip in an environment with python3.11:

```
pip install -e .
```

We recommend installing on a machine with an Nvidia GPU (with Cuda 12):

```
pip install -e .[gpu]
```


## AXIOM

To run our AXIOM agent, run the `main.py` script. The results are dumped to a .csv file and an .mp4 video of the gameplay. When you have wandb set up, results are also pushed to a wandb project called `axiom`.

```
python main.py --game=Explode
```

To see all available configuration options, run `python main.py --help`.

When running on a CPU, or to limit execution time for testing, you can tune down some hyperparameters at the cost of lower average reward, i.e. planning params, bmr samples and number of steps

```
python main.py --game=Explode --planning_horizon 16 --planning_rollouts 16 --num_samples_per_rollout 1 --num_steps=5000 --bmr_pairs=200 --bmr_samples=200
```

We also provide an `example.ipynb` notebook that allows to experiment in a Jupyter notebook and visualize various aspects of the models.
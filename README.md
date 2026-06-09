# Starting code for course project of FAIML - 01VSDWS

Official assignment at [Google Doc](https://docs.google.com/document/d/1AXgLXux3l69vDAPLL-UYD3luFOw3JbyR-pLCS2yuNZk/edit?usp=sharing)

# FAIML Reinforcement Learning Project

Project for the Fundamentals of Artificial Intelligence, Machine and Deep Learning course.

The project studies reinforcement learning algorithms in two control tasks:

- **Part 1:** Hopper-v4 continuous control with REINFORCE and Actor-Critic.
- **Part 2:** PandaPush-v3 robotic manipulation with SAC/PPO and domain randomization.

## Project Overview

This repository contains two main parts:

| Part | Environment | Goal | Methods |
|------|-------------|------|---------|
| Part 1 | MuJoCo Hopper-v4 | Learn locomotion control | REINFORCE, Actor-Critic |
| Part 2 | PandaPush-v3 | Push an object to a target position | SAC, PPO, UDR, ADR |

## Installation
Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```
You can check your installation by launching `python test_random_policy.py`.


## Part 1: Hopper-v4
You can train the policy on the Hopper environment using the `train.py` script in the `part1` folder.

- **REINFORCE**: `python train.py --algorithm reinforce`
```bash
cd part1
python train.py --algorithm reinforce --n_episodes 1000 --gamma 0.99 --lr 0.003
```

- **Actor-Critic**: `python train.py --algorithm actor_critic`
```bash
cd part1
python train.py --algorithm actor_critic --n_episodes 1000 --gamma 0.99 --lr 0.003
```
It is possible to modify the hyperparameters of the training loop by changing the arguments passed to the `train.py` script (i.e. '--lr' for the learning rate).

The best model trained on the Hopper environment can be found in `part1/best_model.pth`.

## Part 2: PandaPush-v3
For the **second part**, you need to install the `panda-gym` package:
```bash
cd part2/panda-gym
pip install -e .
```
You can train the policy on the PandaPush environment using the `train_sb3.py` script in the `part2` folder.
- **SAC**: `python train_sb3.py --algorithm sac`
```bash
cd part2
python train_sb3.py --algorithm sac --env-type source --timesteps 1_000_000 --seed 777
```
- **PPO**: `python train_sb3.py --algorithm ppo`
```bash
cd part2
python train_sb3.py --algorithm ppo --env-type source --timesteps 1_000_000 --seed 777
```
### 1. Local

if you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening.
We highly suggest using Conda to manage the environment.

**Dependencies**
- Run `pip install -r requirements.txt`

Check your installation by launching `python test_random_policy.py`.


### 2. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/)

- Download all files contained in the `colab_template` folder in this repo (inside phase_1 folder).
- Load the `test_random_policy.ipynb` file on [https://colab.research.google.com/](colab) and follow the instructions on it.

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once, to best understand what is going on with the underlying Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.

## 3. Extra step for Push task
To train on the panda-gym task you have to follow these steps first:

```bash
cd part2/panda-gym
pip install -e .
```

## Project structure

```
FAIML-RL-26/
├── README.md
├── requirements.txt
├── part1/ <-- about Hopper
│   ├── agent.py
│   ├── test_random_policy.py
│   ├── train.py
│   └── colab_template/
│       └── test_random_policy.ipynb
└── part2/ <-- about PushTask
    ├── eval_sb3.py
    ├── rand_wrapper.py <-- randomization wrapper for UDR/ADR
    ├── test_random_policy.py
    ├── train_sb3.py
    └── panda-gym/
        └── panda_gym/ (main package)
            └── envs/
                ├── core.py
                ├── panda_tasks.py
                ├── robots/
                │   └── panda.py
                └── tasks/
                    ├── flip.py
                    ├── pick_and_place.py
                    ├── push.py
                    ├── reach.py
                    ├── slide.py
                    └── stack.py
        
```
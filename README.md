# Project of FAIML - 01VSDWS assigned to Group 58 (Campus Luca, Pendin Margherita, Sechi Enrico, Vetrone Antonio)

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

**Note**
We highly suggest using Conda to manage the environment.

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

### Evaluation 
You can evaluate the trained models using the `eval_sb3.py` script in the `part2` folder.
```bash
cd part2
python eval_sb3.py --model-path '[MODEL_PATH]' --render --env-type source
```
The objective of the evaluation is to test the generalization capabilities of the trained policies to unseen environments, by evaluating them on the target environment (i.e. with a different mass for the object to push) and on the source environment to compare the performance. 
You can evaluate the trained models on the target environment by changing the `--model-path` argument with the path where the model trained (source or target) is saved and `--env-type` argument to `target` in the command above.

The `--render` argument allows you to visualize the environment while evaluating the model. You can also specify a delay for rendering to better see the simulation using the `--render-delay` argument in the command above.

### Domain Randomization
To better improve the generalization capabilities of the trained policies, you can also train them with domain randomization. You can choose between Uniform Domain Randomization (UDR) and Adversarial Domain Randomization (ADR) by changing the `--sampling-strategy` argument in the command below.

- **UDR**:
```bash
cd part2
python train_sb3.py --algorithm sac --env-type source --timesteps 1_000_000 --seed 777 --sampling-strategy udr
```
- **ADR**:
```bash
cd part2
python train_sb3.py --algorithm sac --env-type source --timesteps 1_000_000 --seed 777 --sampling-strategy adr
``` 
All the trained models can be found in `part2/models/model.zip`.

## Weights & Biases
The training scripts are set up to log the training process on [Weights & Biases](https://wandb.ai/).

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
    ├── models/ <-- trained models
    │   └── model.zip
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
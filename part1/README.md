To train:
python train.py --n_episodes 1000 --gamma 0.99 --lr 1e-3
python train.py --n_episodes 500 --gamma 0.95 --lr 1e-4

python train.py --n_episodes 1000 --gamma 0.99 --lr 1e-3 --baseline 20
python train.py --n_episodes 1000 --gamma 0.99 --lr 1e-3

# REINFORCE
python train.py --n_episodes 1000 --gamma 0.99 --lr 1e-3 --algorithm reinforce

# REINFORCE with baseline
python train.py --n_episodes 1000 --gamma 0.99 --lr 1e-3 --algorithm reinforce --baseline 20.0

# Actor-Critic
python train.py --n_episodes 1000 --gamma 0.99 --lr 1e-3 --algorithm actor-critic


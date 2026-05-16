To launch trainings:
python train.py --config configs/configs/reinforce_no_baseline_1.json

Infer: 
python infer.py --model-path best_model.pt --render --episodes 3

Infer and Save videos: 
python infer.py --model-path best_model.pt --save-video --episodes 3

synch wandb
wandb sync wandb\latest-run
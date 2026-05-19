import argparse
import os
from rand_wrapper import RandomizationWrapper
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import panda_gym  # noqa: F401 - required so Panda envs are registered


def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"

    env = gym.make(
        "PandaPush-v3",
        render_mode=render_mode,
        reward_type="dense"
    )

    if env_type == "source":
        env = RandomizationWrapper(
            env,
            mass_range=(1.0, 1.0),
            mode="none",
        )

    elif env_type == "target":
        env = RandomizationWrapper(
            env,
            mass_range=(5.0, 5.0),
            mode="none",
        )
    
    model = SAC.load(model_path)  #TODO: load model here

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic) #TODO: get action from the model
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO model zip file (e.g., ppo_panda_push.zip)",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500, 
        help="Number of eval episodes"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--env-type",
        type=str, default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
    )

#SAC was selected as the main algorithm because it achieved significantly better performance than PPO on the PandaPush task.
#Different training durations were tested, and increasing training from 500k to 1M timesteps substantially 
#improved both return and success rate. The source-trained policy achieved good performance on the source environment 
#and slightly lower performance on the target environment, indicating the presence of a domain gap. 
#The best results were obtained by training directly on the target environment, achieving a 92% success rate. 
#These results establish the lower and upper bounds for the subsequent Domain Randomization experiments.

#| Training → Testing | Mean Return | Success Rate |
#| ------------------ | ----------- | ------------ |
#| source → source    | -3.438      | 66%          |
#| source → target    | -3.768      | 62%          | LOWER BOUND
#| target → target    | -1.881      | 92%          | UPPER BOUND

#source→target performs worse than source→source.
#This means the policy trained on light dynamics does not transfer well to heavy dynamics. 
#This will be our lower bound.
#target→target is the better performer since it is trained and tested on the same environment.
#This will be our upper bound.

#The SAC agent trained in the source environment achieved moderate performance when evaluated on the same environment, 
#but performance degraded significantly when tested in the target environment. 
#This indicates that the learned policy overfitted to the source dynamics and did not 
#generalize well to the heavier object dynamics of the target domain. 
#Training directly in the target environment resulted in the best performance, 
#establishing an upper bound for the task. 
#These results motivate the use of Domain Randomization techniques to improve transfer robustness across environments.

#Why do we expect lower performances from the “source→target” configuration w.r.t. the “target→target”?
#The "source→target" configuration involves training the agent in an environment with lighter dynamics (source) 
#and then evaluating it in an environment with heavier dynamics (target). 
#target→target is trained and evaluated in the same environment.
#This mismatch in dynamics can lead to suboptimal performance, as the policy may have learned 
#behaviors that are effective for the source environment but do not translate well to the target environment.

#If higher performances can be reached by training on the target environment directly,
#what prevents us from doing so (in a sim-to-real setting)?
#In a true sim-to-real setting, the “target environment” corresponds to the real world (or the real robot).
#Even though training directly there would give the best performance, it is usually impractical or too expensive.
#The main problems are: 
#Data collection cost, Hardware wear and tear, Safety risks, Time constraints, Need for supervision, Limited scalability.
#Because of these limitations, policies are typically trained in simulation first and then transferred to the real world. 
#The challenge is that the simulator never perfectly matches reality, creating the so-called sim-to-real gap. 
#Domain Randomization is used to reduce this gap by exposing the policy to many different simulated physics 
#conditions during training, making it more robust when deployed in the real environment.

#| Configuration | Mean Return | Success Rate |
#| ------------- | ----------- | ------------ |
#| UDR → source  | -4.221      | 62%          |
#| UDR → target  | -3.878      | 64%          |

#Previously we had a significant drop in performance when evaluating the source-trained policy on the target environment.
#With UDR, the performance drop is much smaller, and we even see a slight improvement in the target environment.
#This suggests that UDR has helped the policy learn more robust behaviors that generalize better across
#different dynamics, reducing the sim-to-real gap.
#The policy is no longer overfitting to the specific mass of the cube in the source environment, and can handle a wider range of masses,
#which is why it performs better in the target environment.
#Uniform Domain Randomization significantly improved transfer robustness between source and target environments. 
#Unlike the baseline source-trained policy, which showed a performance drop when evaluated on the target environment, 
#the UDR-trained policy achieved comparable performance across both domains. 
#This indicates that exposure to randomized dynamics during training helped the policy 
#generalize better to unseen physical conditions.
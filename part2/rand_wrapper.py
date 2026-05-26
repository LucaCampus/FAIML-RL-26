
import gymnasium as gym
import numpy as np

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

        # ADR current adaptive range
        self.mass_min = self.mass_min_limit
        self.mass_max = min(
            self.mass_min_limit +1.0,
            self.mass_max_limit
        )

        # Store recent success history
        self.success_history = []

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):

        if self.mode == "none":
            return self.mass_range[0]  # No randomization, always return the same mass
        
        # Uniform Domain Randomization (UDR)
        elif self.mode == "udr":

            #np.random.uniform samples from a random uniform distribution over [low, high)
            return np.random.uniform(self.mass_min_limit, self.mass_max_limit)
        
        # Adaptive Domain Randomization (ADR)
        elif self.mode == "adr":
            # Sample mass from the current adaptive range
            return np.random.uniform(self.mass_min, self.mass_max)
        else:
            raise NotImplementedError (f'Sampling strategy {self.mode} is not implemented yet.')

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        # ADR curriculum update
        if self.mode == "adr" and done:
            
            succes = float(info.get("is_success", 0.0))
            self.success_history.append(succes)

            # Keep only the recent 20 episodes
            if len(self.success_history) > 20:
                self.success_history.pop(0)

            # Compute recent success rate
            success_rate = np.mean(self.success_history)

            # If agent performs well, expand difficulty by increasing mass range
            if success_rate > 0.3:

                old_mass_max = self.mass_max
                self.mass_max = min(
                    self.mass_max + 0.5,  # Increase max mass by 0.5
                    self.mass_max_limit
                )

                if self.mass_max > old_mass_max:

                    print(
                        f"[ADR] Expanding range -> "
                        f"Expanding mass range to [{self.mass_min:.2f}, {self.mass_max:.2f}]"
                    )

        # Optionally, you can add here extra logic

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):

        new_mass = self._sample_mass()

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

        if self.mode == "adr":

            print(
                f"[ADR] mass={new_mass:.2f} "
                f"current_range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"global_range=[{self.mass_min_limit:.2f},{self.mass_max_limit:.2f}]"
            )
        else:
            print(f"[{self.mode}] mass={new_mass:.2f}"
                  f"range=[{self.mass_min_limit:.2f},{self.mass_max_limit:.2f}]")

        return super().reset(**kwargs)

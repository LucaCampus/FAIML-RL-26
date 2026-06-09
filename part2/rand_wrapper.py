
import gymnasium as gym
import numpy as np

# ADR expands its sampled mass interval after enough successful episodes.
EPISODES_PER_EVAL = 20
SUCCESS_THRESH = 0.7
MAX_INCREMENT = 0.5

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

        # Global mass limits used by UDR and as the maximum ADR can expand to.
        self.mass_min_limit, self.mass_max_limit = mass_range
        
        # ADR starts from a narrower range and expands the upper bound over time.
        self.mass_min = self.mass_min_limit
        self.mass_max = min(
            self.mass_min_limit + 1.0,
            self.mass_max_limit
        )

        self.success_history = []

    def _sample_mass(self):

        if self.mode == "none":
            # Keep the environment-defined source/target mass unchanged.
            return None
        
        elif self.mode == "udr":
            # Uniform Domain Randomization samples from the full fixed range.
            return np.random.uniform(
                self.mass_min_limit,
                self.mass_max_limit,
            )
        
        elif self.mode == "adr":
            # Automatic Domain Randomization samples from the current adaptive range.
            return np.random.uniform(
                self.mass_min,
                self.mass_max
            )
        else:
            raise NotImplementedError(
                f"Sampling method {self.mode} is not implemented yet!"
            )

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if self.mode == "adr" and done:
            # Use recent episode successes to decide when to make ADR harder.
            success = float(info.get("is_success", 0.0))
            self.success_history.append(success)

            if len(self.success_history) >= EPISODES_PER_EVAL:
                self.success_history.pop(0)

            success_rate = np.mean(self.success_history)

            if success_rate > SUCCESS_THRESH:
                old_mass_max = self.mass_max

                self.mass_max = min(
                    self.mass_max + MAX_INCREMENT,
                    self.mass_max_limit
                )

                if self.mass_max > old_mass_max:
                    print(
                        f"[ADR] Expanding range -> [{self.mass_min:.2f},{self.mass_max:.2f}]"
                    )


        

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):

        new_mass = self._sample_mass()

        if new_mass is not None:

            # Change only the pushed object dynamics; robot and target marker are left unchanged.
            sim = self.env.unwrapped.task.sim # type: ignore
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

                print(
                    f"[{self.mode}] mass={new_mass:.2f} "
                    f"range=[{self.mass_min_limit:.2f},{self.mass_max_limit:.2f}]"
                )

        return super().reset(**kwargs)

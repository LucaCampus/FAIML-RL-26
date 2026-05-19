import gymnasium as gym
import numpy as np

#Create a custom wrapper around the enviroment.
#A wrapper modifies the env without rewriting the original one, so that the agent interacts directly with the wrapper.
class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """

    #Initialize the wrapper.
    def __init__(
        self,
        env,
        #Defines minimimum and maximum possible mass
        mass_range=(1.0, 1.0),
        #Randomization mode
        mode="none",
    ):
        super().__init__(env)

        #Store parameters.
        self.mode = mode
        self.mass_range = mass_range

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

        # ADR current adaptive range
        self.mass_min = self.mass_min_limit
        self.mass_max = min(
            self.mass_min_limit + 1.0,
            self.mass_max_limit
        )

        # Store recent success history
        self.success_history = []

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):

        #If there isn't any randomization we always return a fixed mass.
        if self.mode == "none":
            return self.mass_range[0]

        #Uniform Domain Randomization
        elif self.mode == "udr":

            #np.random.uniform samples from a random uniform distribution.
            return np.random.uniform(
                self.mass_min_limit,
                self.mass_max_limit
            )

        #Automatic Domain Randomization
        elif self.mode == "adr":
            
            return np.random.uniform(
                self.mass_min,
                self.mass_max
            )

        else:
            raise NotImplementedError(
                f"Sampling strategy '{self.mode}' is not implemented yet.")


    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        # ADR curriculum update
        if self.mode == "adr" and done:

            success = float(info.get("is_success", 0.0))

            self.success_history.append(success)

            # Keep only recent episodes
            # How many recent episodes are used to estimate performance.
            if len(self.success_history) >= 20:
                self.success_history.pop(0)

            # Compute recent success rate
            success_rate = np.mean(self.success_history)

            # If agent performs well, expand difficulty
            if success_rate > 0.5:

                self.mass_max = min(
                    self.mass_max + 0.5,
                    self.mass_max_limit
                )

                print(
                    f"[ADR] Expanding range -> "
                    f"[{self.mass_min:.2f}, {self.mass_max:.2f}]"
                )

        #Wrapper does not modify behavior during step.
        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------
    #Called at the beggining of the episode so each episode can use different physics.
    def reset(self, **kwargs):

        #Gets mass according to strategy.
        new_mass = self._sample_mass()

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            #It modifies physical properties inside PyBullet.
            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

        # Print current configuration
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

        #Finally resets the underlying environment.
        return super().reset(**kwargs)


#This wrapper is the ENTIRE sim-to-real mechanism, it changes the env physics without changing the task, the robot,
#or the policy.

# Uniform Domain Randomization (UDR):
# UDR improves robustness by randomly changing environment parameters during training.
# In this project, the cube mass is sampled uniformly from a predefined range at the beginning
# of each episode. This forces the agent to learn behaviors that work across multiple dynamics
# instead of overfitting to a single environment configuration.

# Automatic Domain Randomization (ADR):
# ADR extends UDR by adapting the randomization range automatically during training.
# The environment starts with easier conditions and progressively increases the variability
# as the agent improves. This creates a curriculum learning effect that helps the policy
# gradually become robust to a wider range of physical dynamics.
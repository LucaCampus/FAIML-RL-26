
import gymnasium as gym

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

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):

        #If there isn't any randomization we always return a fixed mass.
        if self.mode == "none":
            return self.mass_range[0]

        else:
            raise NotImplementedError(
                f"Sampling strategy '{self.mode}' is not implemented yet.")


    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        #Wrapper does not modify behavior during step.
        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------
    #Called at the beggining of the episode so each episode can use different physics.
    def reset(self, **kwargs):

        #Gets mass according to strategy.
        new_mass = self._sample_mass() #TODO: sample new mass

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            #It modifies physical properties inside PyBullet.
            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

        print(
            f"[{self.mode}] mass={new_mass:.2f} "
            f"range=[{self.mass_min_limit:.2f},{self.mass_max_limit:.2f}]"
            )

        #Finally resets the underlying environment.
        return super().reset(**kwargs)


#This wrapper is the ENTIRE sim-to-real mechanism, it changes the env physics without changing the task, the robot,
#or the policy.
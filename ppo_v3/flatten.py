"""Wrapper for flattening observations of an environment."""
import gym
import gym.spaces as spaces


class FlattenObservation(gym.ObservationWrapper):
    """Observation wrapper that flattens the observation.
    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env: gym.Env, num_envs=1):
        """Flattens the observations of an environment.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self._num_envs = num_envs
        #print("obs space", env.observation_space)
        #observation_space = gym.spaces.Tuple([env.observation_space for _ in range(num_envs)])
        #print(observation_space)
        self.observation_space = spaces.flatten_space(env.observation_space)
        #print("obs space", self.observation_space)


    def observation(self, observation):
        """Flattens an observation.
        Args:
            observation: The observation to flatten
        Returns:
            The flattened observation
        """
        observation = spaces.flatten(self.env.observation_space, observation).reshape(self._num_envs, -1)
        return observation
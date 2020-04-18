from gym import spaces
from gym.utils import seeding
import gym
import math
import numpy as np


G_EARTH = 9.807
MAX_RANGE_M224 = 3490.0
MAX_RADIUS_M224 = 25.0


class FireSolution:
    # M224 60mm: 20-25 m; 20 rpm; 70–3490m range; 241 m/s muzzle velocity

    def __init__ (self, _g=G_EARTH, _range=MAX_RANGE_M224, _radius=MAX_RADIUS_M224):
        self.g = _g
        self.range = _range
        self.radius = _radius
        self.velocity = math.sqrt(self.range * self.g)


    @classmethod
    def deg_to_rad (cls, degree):
        return degree * math.pi / 180.0


    def calc_dist (self, theta):
        return self.velocity**2.0 / self.g * math.sin(theta)


class Projectile_v0 (gym.Env):
    metadata = {"render.modes": ["human"]}
    reward_range = (-100.0, 100.0)


    def __init__ (self):
        self.fire = FireSolution()
        self.action_space = spaces.Box(low=np.float32(0), high=np.float32(90), shape=(1,))
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(self.fire.range), shape=(1,))
        self.np_random = None
        self.reset()


    def reset (self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.seed()
        self.state = round(self.np_random.random() * self.fire.range)

        self.reward = -100.0
        self.done = 0
        self.info = {}

        return self.state


    def step (self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : array[float]

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.done == 1:
            print("game over")
            return [self.state, self.reward, self.done, self.info]

        else:
            degree = action[0]
            theta = self.fire.deg_to_rad(degree)
            dist = self.fire.calc_dist(theta)
            delta = abs(self.state - dist)

            self.info["theta"] = theta
            self.info["dist"] = dist
            self.info["delta"] = delta

            self.render()

        if delta <= self.fire.radius:
            self.reward = 100.0
            self.done = 1;
            print("win")
        else:
            # reward is the "nearness" of the blast destroying the target
            self.reward = 100.0 * (self.fire.range - delta) / self.fire.range

        return [self.state, self.reward, self.done, self.info]


    def render (self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        print("location:", self.state)


    def close (self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass


    def seed (self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

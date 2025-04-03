import unittest
import numpy as np
import gym
from gym import register

register(
    id='2denv-v0', # Environment ID.
    entry_point='environment:envGym', # Environment ID.
)
class TestEnv(unittest.TestCase):
    # Setup function to initialize the environment before each test case.
    def setUp(self):
        self.env = gym.make('2denv-v0')
        # Create an instance of the custom environment.
        self.env.reset()
        # Reset the environment to its initial state.

    # Teardown function to clean up after each test case.
    def tearDown(self):

        self.env.close()

    # Test for the environment reset functionality.
    def test_reset(self):
        print("Running test_reset...")
        initial_state = self.env.reset()
        # Test for the environment reset functionality.
        print(f"Initial state: {initial_state}")
        self.assertIsNotNone(initial_state)
        # Check that the reset state is not None.
        self.assertEqual(len(initial_state), 10)
        # Ensure the reset state has the correct size.
        self.assertTrue(isinstance(initial_state, np.ndarray))
        # Verify the reset state is a NumPy array.

    # Test for the step functionality of the environment.
    def test_step(self):
        print("Running test_step...")
        action = [0.5, -0.5, 0.5, -0.5]
        # Define a sample action for testing.
        state, reward, terminated, info = self.env.step(action)
        # Perform a step in the environment using the sample action.
        print(f"State: {state}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
        self.assertIsNotNone(state)
        # Ensure the resulting state is not None.
        self.assertEqual(len(state), 10)
        # Check that the resulting state has the correct size.
        self.assertTrue(isinstance(state, np.ndarray))
        # Verify the resulting state is a NumPy array.
        self.assertTrue(isinstance(reward, float))
        # Ensure the reward is a floating-point number.
        self.assertTrue(isinstance(terminated, bool))
        # Verify the terminated flag is a boolean.
        self.assertTrue(isinstance(info, dict))
        # Check that the info object is a dictionary.

    def test_observation_space(self):
        print("Running test_observation_space...")
        print(f"Observation Space: {self.env.observation_space}")
        self.assertIsNotNone(self.env.observation_space)
        self.assertTrue(isinstance(self.env.observation_space, gym.spaces.Box))
        self.assertEqual(self.env.observation_space.shape, (10,))

    # Test for the observation space of the environment.
    def test_action_space(self):
        print("Running test_action_space...")
        print(f"Action Space: {self.env.action_space}")
        self.assertIsNotNone(self.env.action_space)
        # Ensure the observation space is not None.
        self.assertTrue(isinstance(self.env.action_space, gym.spaces.Box))
        # Verify it is a Box space.
        self.assertEqual(self.env.action_space.shape, (4,))
        # Check the shape of the observation space.

    # Test for the action space of the environment.
    def test_render(self):
        print("Running test_render...")
        try:
            self.env.render(mode='human') # Test rendering in human-readable mode.
            print("Rendered in 'human' mode successfully.")
        except Exception as e:
            print(f"Error rendering in 'human' mode: {e}")

        try:
            self.env.render(mode='rgb_array')  # Test rendering in RGB array mode.
            print("Rendered in 'rgb_array' mode successfully.")
        except Exception as e:
            print(f"Error rendering in 'rgb_array' mode: {e}")

        try:
            self.env.render(mode='speed') # Test rendering in speed mode.
            print("Rendered in 'speed' mode successfully.")
        except Exception as e:
            print(f"Error rendering in 'speed' mode: {e}")

if __name__ == '__main__':
    unittest.main()


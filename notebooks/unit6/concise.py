import os

import gymnasium as gym
import panda_gym

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from pyvirtualdisplay import Display

env_id = "PandaReachDense-v3"
#env_id = "PandaPickAndPlace-v3"
model_file = 'panda-result'
env_file = 'vec_normalize.pkl'

# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation
print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action


# Create the env
env = make_vec_env(env_id, n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = A2C(policy = "MultiInputPolicy",
            env = env,
            verbose=1)

'''
model.learn(1_000_000)
# Save the model and VecNormalize statistics when saving the agent
model.save(model_file)
env.save(env_file)
'''

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make(env_id)])
eval_env = VecNormalize.load(env_file, eval_env)

# We need to override the render_mode
eval_env.render_mode = "rgb_array"

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load(model_file)
mean_reward, std_reward = evaluate_policy(model, eval_env)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Show
virtual_display = Display(visible=1, size=(1400, 900))
virtual_display.start()

obs = eval_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render("human")

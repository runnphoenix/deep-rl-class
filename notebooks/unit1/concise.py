# Virtual display
from pyvirtualdisplay import Display

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym

# First, we create our environment called LunarLander-v2
env = gym.make("LunarLander-v2")
# Then we reset this environment
observation, info = env.reset()

# We added some parameters to accelerate the training
model_name = "ppo-LunarLander-v2"
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

# Train it for 1,000,000 timesteps
#model.learn(total_timesteps=1200000)

# Save the model
#model.save(model_name)

# Eval model performance
eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Load model
model = PPO.load(model_name)

# Show
virtual_display = Display(visible=1, size=(1400, 900))
virtual_display.start()

vec_env = make_vec_env("LunarLander-v2", n_envs=4)
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

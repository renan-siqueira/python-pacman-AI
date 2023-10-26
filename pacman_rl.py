import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import torch

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using: {device}")


def train_pacman():
    env_id = 'MsPacmanNoFrameskip-v4'
    env = gym.make(env_id)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)
    env = DummyVecEnv([lambda: env])

    model = DQN('CnnPolicy', env, verbose=1, buffer_size=50000)
    model.learn(total_timesteps=1000000)
    model.save("dqn_pacman")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Recompensa média: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model


def test_pacman(model):
    env = gym.make('MsPacman-v0')

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()

if __name__ == '__main__':
    trained_model = train_pacman()
    test_pacman(trained_model)

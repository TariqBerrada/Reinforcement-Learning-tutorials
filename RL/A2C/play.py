import os, time, imageio, argparse
import numpy as np
from atari_wrappers import *
from a2c import Agent
from neural_network import CNN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help = 'environment ID', default = 'BreakoutNoFrameskip-v4')
    return parser.parse_args()

def get_agent(env, n_steps = 5, n_stack = 1, total_timesteps= int(80e6), vf_coef = .5, ent_coef = .01, max_grad_norm = .5, lr = 7e-4, epsilon = 1e-5, alpha = .99):
    agent = Agent(Network = CNN, ob_space = env.observation_space, ac_space = env.action_space, n_envs = 1, n_steps = n_steps, n_stack = n_stack, ent_coef = ent_coef, vf_coef = vf_coef, max_grad_norm = max_grad_norm, lr = lr, alpha = alpha, epsilon = epsilon, total_timesteps= total_timesteps)
    return agent

def main():
    env_id = get_args().env
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack = True, clip_rewards = False, episode_life = True)
    env = Monitor(env)

    agent = get_agent(env)

    save_path = os.path.join('models', env_id + '.save')
    agent.load(save_path)

    obs = env.reset()
    renders = []
    while = True:
        obs = np.expand_dims(obs.__array__(), axis = 0)
        a, v = agent.step(obs)
        obs, reward, done, info = env.step(a)
        env.render()
        if done:
            print(info)
            env.reset()

if __name__ == '__main__':
    main()
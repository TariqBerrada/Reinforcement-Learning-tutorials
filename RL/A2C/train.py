from subproc_vec_env import SubprocVecEnv
from atari_wrappers import *
from neural_network import CNN
from a2c import learn

import os, gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = 'models'
SEED = 0

def train(env_id, num_timesteps, num_cpu):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(SEED + rank)
            gym.logger.setLevel(logging.WARN)
            env = wrap_deepmind(env)
            env = Monitor(env, rank)
            return env
        return _thunk
    
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    learn(CNN, env, SEED, total_timesteps = int(num_timesteps*1.1))
    env.close()
    pass

env = 'BreakoutNoFrameskip-v4'
steps = int(80e6)
n_env = 16

def main():
    os.makedirs(MODEL_PATH, exist_ok = True)
    train(env, steps, num_cpu = n_env)

if __name__ == '__main__':
    main()
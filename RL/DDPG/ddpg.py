import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import gym, joblib
import matplotlib.pyplot as plt
from datetime import datetime

def ANN(x, layer_sizes, hidden_activation = tf.nn.relu, output_activation = None):
    for h in layer_sizes[:-1]:
        x = tf.layers.dense(x, units = h, activation = hidden_activation)
    return tf.layers.dense(x, layer_sizes[-1], activation = output_activation)

def get_variables(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def CreateNetworks(s, a, n_actions, action_max, hidden_sizes = (300,), hidden_activation = tf.nn.leaky_relu, output_activation= tf.tanh):
    with tf.variable_scope('mu'):
        mu = action_max*ANN(s, list(hidden_sizes) + [n_actions], hidden_activation, output_activation)
    with tf.variable_scope('q'):
        _input = tf.concat([s, a], axis = -1)
        q = tf.squeeze(ANN(_input, list(hidden_sizes) + [1], hidden_activation, None), axis = 1)
    with tf.variable_scope('q', reuse = True):
        _input = tf.concat([s, mu], axis = 1)
        q_mu = tf.squeeze(ANN(_input, list(hidden_sizes) +[1], hidden_activation, None), axis = 1)
    return mu, q, q_mu

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype = np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype = np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype = np.float32)
        self.rews_buf = np.zeros(size, dtype = np.float32)
        self.done_buf = np.zeros(size, dtype = np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1)%self.max_size
        self.size = min(self.size+1, self.max_size)
    
    def sample_batch(self, batch_size = 32):
        idxs = np.random.randint(0, self.size, size = batch_size)
        return dict(
            s = self.obs1_buf[idxs],
            s2 = self.obs2_buf[idxs],
            a = self.acts_buf[idxs],
            r = self.rews_buf[idxs],
            d = self.done_buf[idxs]
        )

def ddpg(env_fn, ac_kwargs = dict(), seed = 0, save_folder = None, n_train_episodes = 100, test_agent_every = 25, replay_size = int(1e6), gamma = .99, decay = .995, mu_lr = 1e-3, q_lr = 1e-3, batch_size = 100, start_steps = 10000, action_noise = .1, max_episode_length = 1000):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    if save_folder is not None:
        test_env = gym.wrappers.Monitor(test_env, save_folder)
    
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # min value has the same magnitude as the min value
    # actions have the same range.
    action_max = env.action_space.high[0]

    X = tf.placeholder(dtype = tf.float32, shape = (None, n_states)) # state
    A = tf.placeholder(dtype = tf.float32, shape = (None, n_actions)) # action
    X2 = tf.placeholder(dtype = tf.float32, shape = (None, n_states)) # next state
    R = tf.placeholder(dtype = tf.float32, shape = (None,)) # reward
    D = tf.placeholder(dtype = tf.float32, shape = (None,)) # done flag

    with tf.variable_scope('main'):
        mu, q, q_mu = CreateNetworks(X, A, n_actions, action_max, **ac_kwargs)
    with tf.variable_scope('target'):
        _, _, q_mu_target = CreateNetworks(X2, A, n_actions, action_max, **ac_kwargs)

    replay_buffer = ReplayBuffer(obs_dim = n_states, act_dim = n_actions, size = replay_size)
    q_target = tf.stop_gradient(R+gamma*(1-D)*q_mu_target)
    mu_loss = -tf.reduce_mean(q_mu)
    q_loss = tf.reduce_mean((q - q_target)**2)

    mu_optimizer = tf.train.AdamOptimizer(learning_rate = mu_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate = q_lr)
    mu_train_op = mu_optimizer.minimize(mu_loss, var_list = get_variables('main/mu'))
    q_train_op = q_optimizer_op = q_optimizer.minimize(q_loss, var_list = get_variables('main/q'))

    target_update = tf.group(
        [tf.assign(v_target, decay*v_target + (1 - decay)*v_main) for v_main, v_target in zip(get_variables('main'), get_variables('target'))]
    )

    target_init = tf.group(
        [tf.assign(v_targ, v_main) for v_main, v_targ in zip(get_variables('main'), get_variables('target'))]
    )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(s, noise_scale):
        a = sess.run(mu, feed_dict = {X:s.reshape(1, -1)})[0]
        a += noise_scale*np.random.randn(n_actions)
        return np.clip(a, -action_max, action_max)

    test_returns = []

    def test_agent(n_episodes = 5):
        t0 = datetime.now()
        n_steps = 0
        for j in range(n_episodes):
            s, episode_return, episode_length = test_env.reset(), 0, 0
            while not (d or (episode_length == max_episode_length)):
                test_env.render()
                s, r, d, _ = test_env.step(get_action(s, 0))
                episode_return += r
                episode_length += 1
                n_steps += 1
            print('test return : ', episode_return, 'episode_length : ', episode_length)
            test_returns.append(episode_return)

    returns = []
    q_losses = []
    mu_losses = []
    n_steps = 0

    for i_episode in range(n_train_episodes):
        s, episode_return, episode_length, d = env.reset(), 0, 0, False
        while not (d or (episode_length == max_episode_length)):
            if n_steps > start_steps:
                a = get_action(s, action_noise)
            else:
                a = env.action_space.sample()
            
            n_steps += 1
            if n_steps == start_steps:
                print('using agent actions now')

            s2, r, d, _ = env.step(a)
            episode_return += r
            episode_length += 1

            d_store = False if episode_length == max_episode_length else d
            # add experience to the experience replay buffer.
            replay_buffer.store(s, a, r, s2, d_store)

            # assign next state to current state
            s = s2

            # Perform updates
            for _ in range(episode_length):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {
                    X: batch['s'],
                    X2: batch['s2'],
                    A: batch['a'],
                    R: batch['r'],
                    D: batch['d']
                }

                # Q-network update.
                ql, _, _ = sess.run([q_loss, q, q_train_op], feed_dict)
                q_losses.append(ql)

                # Policy update
                mul, _, _ = sess.run([mu_loss, mu_train_op, target_update], feed_dict)
                mu_losses.append(mu)

            print('episode : %d | return : %.4f | length : %d'%(i_episode+1, episode_return, episode_length))
        returns.append(episode_return)

        # test agent
        if i_episode > 0 and i_episode%test_agent_every == 0:
            test_agent()
        
    _dict = {'train' : returns, 'test' : test_returns, 'q_losses' : q_losses, 'mu_losses' : mu_losses}
    joblib.dump(_dict, 'ddpg_results.npz')
            # np.savez('ddpg_results.npz', train = returns, test = test_returns, q_losses = q_losses, mu_losses = mu_losses)
        
def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum()/(i-start+1))
    return y

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type = str, default = 'Pendulum-v0')
    parser.add_argument('--hidden_layer_sizes', type = int, default = 300)
    parser.add_argument('--num_layers', type = int, default = 1)
    parser.add_argument('--gamma', type = float, default = .99)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--n_train_episodes', type = int, default = 200)
    parser.add_argument('--save_folder', type = str, default = 'ddpg_monitor')
    args = parser.parse_args()

    ddpg(
        lambda : gym.make(args.env),
        ac_kwargs = dict(hidden_sizes = [args.hidden_layer_sizes]*args.num_layers),
        gamma = args.gamma,
        seed = args.seed,
        save_folder = args.save_folder,
        n_train_episodes = args.n_train_episodes,
    )
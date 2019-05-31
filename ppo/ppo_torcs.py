import os

import numpy as np
import torch
from model.ppo import generate_trajectory, ppo_update
from torch.nn import MSELoss
from torch.optim import Adam
from gym_torcs import TorcsEnv

from model.net import MLPPolicy

def train(device):
    # hyper-parameters
    coeff_entropy = 0.00001
    lr = 5e-4
    mini_batch_size = 64
    horizon = 2048
    nupdates = 10
    nepoch = 5000
    clip_value = 0.2
    train = True
    render = False
    # initialize env
    env = TorcsEnv(port=3101, path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")
    insize = env.observation_space.shape[0]
    outsize = env.action_space.shape[0]

    policy = MLPPolicy(insize, action_space = outsize)
    policy.to(device)
    if os.path.exists('policy.pth'):
        policy.load_state_dict(torch.load('policy.pth', map_location = device))
        print('Loading complete!')
    if train:
        optimizer = Adam(lr=lr, params=policy.parameters())
        mse = MSELoss()

        # start training
        for e in range(nepoch):
            # generate trajectories
            relaunch = e%100 == 0
            observations, actions, logprobs, returns, values, rewards = \
                generate_trajectory(env, policy, horizon, is_render=render,
                                    obs_fn=None, progress=True, device=device, is_relaunch = relaunch)
            print('Episode %s reward is %s' % (e, rewards.sum()))
            memory = (observations, actions, logprobs, returns[:-1], values)
            # update using ppo
            policy_loss, value_loss, dist_entropy =\
                ppo_update(
                    policy, optimizer, mini_batch_size, memory, nupdates,
                    coeff_entropy=coeff_entropy, clip_value=clip_value, device=device
                )
            print('\nEpisode: {}'.format(e))
            print('Total reward {}'.format(rewards.sum()))
            print('Entropy', dist_entropy)
            print('Policy loss', policy_loss)
            print('Value loss', value_loss)
            if np.mod(e+1, 10) == 0:
                print("saving model")
                torch.save(policy.state_dict(), 'policy.pth')


if __name__ == "__main__":
    train("cpu")
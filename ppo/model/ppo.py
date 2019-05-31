import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def calculate_returns(rewards, dones, last_value, gamma=0.99):
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(rewards.shape[0])):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns

def ppo_update(policy, optimizer, batch_size, memory, nupdates,
               coeff_entropy=0.02, clip_value=0.2, writer=None, device="cpu"):
    obs, actions, logprobs, returns, values = memory
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()

    for update in range(nupdates):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advantages.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obs[index])).float().to(device)
            sampled_actions = Variable(torch.from_numpy(actions[index])).float().to(device)
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().to(device)
            sampled_returns = Variable(torch.from_numpy(returns[index])).float().to(device)
            sampled_advs = Variable(torch.from_numpy(advantages[index])).float().to(device)

            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_returns = sampled_returns.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_returns)

            loss = policy_loss + value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar('ppo/value_loss', value_loss.item())
                writer.add_scalar('ppo/policy_loss', policy_loss.item())
                writer.add_scalar('ppo/entropy', dist_entropy.item())
    return value_loss.item(), policy_loss.item(), dist_entropy.item()


def generate_trajectory(env, policy, max_step, obs_fn=None, progress=False,
                        is_render=False, device="cpu", is_relaunch = False):
    """generate a batch of examples using policy"""
    nstep = 0
    obs = env.reset(relaunch = is_relaunch, render = is_render, sampletrack = False)
    print("trajectory started:\n")
    done = False
    observations, rewards, actions, logprobs, dones, values = [], [], [], [], [], []
    while not (nstep == max_step):
        if done:
            break
        if obs_fn is not None:
            obs = obs_fn(obs)
        obs = Variable(torch.from_numpy(obs[np.newaxis])).float().to(device)

        value, action, logprob, mean = policy(obs)
        value, action, logprob = value.data.cpu().numpy()[0], action.data.cpu().numpy()[0], \
                                 logprob.data.cpu().numpy()[0]
        #try new action method
        action[1] = min((action[1]+1)/2, 1)
        action[1] = max(action[1], 0.5) 
        next_obs, reward, done, _ = env.step(np.concatenate([action[:2], [-1]]))
        #next_obs, reward, done, _ = env.step(action)
        observations.append(obs.data.cpu().numpy()[0])
        rewards.append(reward)
        logprobs.append(logprob)
        dones.append(done)
        values.append(value[0])
        actions.append(action)

        obs = next_obs
        nstep += 1
        if progress:
            print('\r{}/{}'.format(nstep, max_step), flush=True, end='')

    if done:
        last_value = 0.0
    else:
        if obs_fn is not None:
            obs = obs_fn(obs)
        obs = Variable(torch.from_numpy(obs[np.newaxis])).float().to(device)
        value, action, logprob, mean = policy(obs)
        last_value = value.data[0][0]
    observations = np.asarray(observations)
    rewards = np.asarray(rewards)
    logprobs = np.asarray(logprobs)
    dones = np.asarray(dones)
    values = np.asarray(values)
    actions = np.asarray(actions)
    returns = calculate_returns(rewards, dones, last_value)
    return observations, actions, logprobs, returns, values, rewards

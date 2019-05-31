import numpy as np
import torch
import random
from collections import namedtuple
from collections import defaultdict
from agent.ddpg import Ddpg
from agent.simple_network import ActorNetwork
from agent.simple_network import CriticNetwork
from agent.random_process import OrnsteinUhlenbeckProcess
from gym_torcs import TorcsEnv

def train(device):
    env = TorcsEnv(port=3101, path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")
    insize = env.observation_space.shape[0]
    outsize = env.action_space.shape[0]

    hyperparams = {
                "lrvalue": 0.001,
                "lrpolicy": 0.001,
                "gamma": 0.985,
                "episodes": 3000,
                "buffersize": 300000,
                "tau": 0.01,
                "batchsize": 32,
                "start_sigma": 0.9,
                "end_sigma": 0.1,
                "theta": 0.15,
                "maxlength": 100000,
                "clipgrad": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    datalog = defaultdict(list)
    
    valuenet = CriticNetwork(insize, outsize)
    policynet = ActorNetwork(insize)
    
    #load model
    print("loading model")
    try:
        policynet.load_state_dict(torch.load('policynet.pth', map_location = device))
        policynet.eval()
        valuenet.load_state_dict(torch.load('valuenet.pth', map_location = device))
        valuenet.eval()
        print("model load successfully")
    except:
        print("cannot find the model")    
    
    agent = Ddpg(valuenet, policynet, buffersize=hyprm.buffersize)
    agent.to(device)

    for eps in range(hyprm.episodes):
        state = env.reset(relaunch=eps%100 == 0, render=True, sampletrack=False)
        epsisode_reward = 0
        episode_time = []
        sigma = (hyprm.start_sigma-hyprm.end_sigma)*(max(0, 1-eps/hyprm.episodes)) + hyprm.end_sigma
        randomprocess = OrnsteinUhlenbeckProcess(hyprm.theta, sigma, outsize)
        for i in range(hyprm.maxlength):
            torch_state = agent._totorch(state, torch.float32).view(1, -1)
            action, value = agent.act(torch_state)
            action = randomprocess.noise() + action.to("cpu").squeeze()
            action.clamp_(-1, 1)
            action[1] = (action[1]+1)/2
            next_state, reward, done, _ = env.step(np.concatenate([action[:2], [-1]]))
            agent.push(state, action, reward, next_state, done)
            epsisode_reward += reward

            if len(agent.buffer) > hyprm.batchsize:
                value_loss, policy_loss = agent.update(hyprm.gamma, hyprm.batchsize, hyprm.tau, hyprm.lrvalue, hyprm.lrpolicy, hyprm.clipgrad)
                if random.uniform(0, 1) < 0.01:
                    datalog["td error"].append(value_loss)
                    datalog["avearge policy value"].append(policy_loss)

            if done:
                break
            state = next_state
        episode_time.append(i)
        datalog["total reward"].append(epsisode_reward)
        
        avearage_reward = torch.mean(torch.tensor(datalog["total reward"][-20:])).item()
        avg_td_error = torch.mean(torch.tensor(datalog["td error"][-20:])).item()
        avg_policy_val = torch.mean(torch.tensor(datalog["avearge policy value"][-20:])).item()
        avg_time = np.mean(episode_time).item()
        print("\r Processs percentage: {:2.1f}%, Average reward: {:2.3f}".format(eps/hyprm.episodes*100, avearage_reward), end="", flush=True)
        print(" td_error: {:2.3f}, policy_val: {:2.3f}, avg_time: {}".format(avg_td_error, avg_policy_val, avg_time), end="", flush=True)
        if np.mod(eps+1, 10) == 0:
            print("saving model")
            torch.save(agent.policynet.state_dict(), 'policynet.pth')
            torch.save(agent.valuenet.state_dict(), 'valuenet.pth')
    print("")



if __name__ == "__main__":
    train("cpu")
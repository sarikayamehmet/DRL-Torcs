import torch
from .replaybuffer import UniformBuffer
from .replaybuffer import Transition
import logging
import random
from copy import deepcopy
from itertools import chain

class Ddpg(torch.nn.Module):

    def __init__(self, valuenet,
                    policynet,
                    buffer=None,
                    buffersize=10000,
                    logger_level=logging.WARNING):
        super().__init__()

        self.valuenet = valuenet
        self.policynet = policynet
        self.targetvaluenet = deepcopy(valuenet)
        self.targetpolicynet = deepcopy(policynet)
        
        self.buffer = buffer or UniformBuffer(buffersize, logger_level=logger_level)

        self.logger = logging.getLogger(__name__ + __class__.__name__)
        self.logger.setLevel(logger_level)

        self.device = "cpu"
        self.opt_value = torch.optim.Adam(self.valuenet.parameters())
        self.opt_policy = torch.optim.Adam(self.policynet.parameters())

    def act(self, state, withvalue=False):
        """ 
            Args:
                - state: Batch size 1 torch tensor.
        """
        self.eval()
        if state.shape[0] != 1:
            raise ValueError("Batch size of the state must be 1! Instead: {}".format(state.shape[0]))

        value = None
        with torch.no_grad():
            action = self.policynet(state)
            if withvalue:
                value = self.valuenet(state, action).item()
        action = action.squeeze()

        return (action, value)

    def td_error(self, gamma, batch):

        with torch.no_grad():
            target_action = self.targetpolicynet(batch.next_state)
            target_value = self.targetvaluenet(batch.next_state, target_action)
        
        current_value = self.valuenet(batch.state, batch.action)
        next_value = target_value*(1 - batch.terminal)*gamma + batch.reward
        td_loss = torch.nn.functional.smooth_l1_loss(current_value, next_value)
        return td_loss

    def policy_loss(self, batch):

        action = self.policynet(batch.state)
        value = self.valuenet(batch.state, action)
        return -torch.mean(value)

    def update_target(self, tau):
        for net, tarnet in zip((self.policynet, self.valuenet),
                                 (self.targetpolicynet, self.targetvaluenet)):
            
            for param, tparam in zip(net.parameters(), tarnet.parameters()):
                tparam.data += tau*(param.data - tparam.data)
            
    def clip_grad(self, parameters):
        for param in parameters:
            param.grad.data.clamp_(-1, 1)
        self.logger.debug("Gradient is clipped!")

    def update(self, gamma, batchsize, tau, lrvalue=None, lrpolicy=None, gradclip=False):
        self.train()
        for lr, opt in zip((lrpolicy, lrvalue),
                             (self.opt_policy, self.opt_value)):
            if lr:
                for g in opt.param_groups:
                    g["lr"] = lr

        batch = self.buffer.sample(batchsize)
        batch = self._batchtotorch(batch)

        # ----  Value Update --------
        self.opt_value.zero_grad()
        loss_value = self.td_error(gamma, batch)
        loss_value.backward()
        if gradclip:
            self.clip_grad(self.valuenet.parameters())
        self.opt_value.step()

        # ---  Policy Update -------
        self.opt_policy.zero_grad()
        loss_policy = self.policy_loss(batch)
        loss_policy.backward()
        if gradclip:
            self.clip_grad(self.policynet.parameters())
        self.opt_policy.step()

        # ----- Target Update -----
        self.update_target(tau)

        return (loss_value.item(), -loss_policy.item()) 

    def push(self, state, action, reward, next_state, terminal):
        self.buffer.push(**dict(state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                terminal=terminal))

    def _batchtotorch(self, batch):
        state = self._totorch(batch.state, torch.float32)
        action = self._totorch(batch.action, torch.float32)
        next_state = self._totorch(batch.next_state, torch.float32)
        terminal = self._totorch(batch.terminal, torch.float32).view(-1, 1)
        reward = self._totorch(batch.reward, torch.float32).view(-1, 1)
        return Transition(state, action, reward, next_state, terminal)

    def _totorch(self, container, dtype):
        if isinstance(container[0], torch.Tensor):
            tensor = torch.stack(container)
        else:
            tensor = torch.tensor(container, dtype=dtype)
        return tensor.to(self.device)

    def to(self, device):
        self.device = device
        self.logger.debug("Device is changed to: {}!".format(str(device)))
        super().to(device)

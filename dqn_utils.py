import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from types import SimpleNamespace

from ptan import ptan
from typing import List

class DQN(nn.Module):
    def __init__(self, obs_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)
    

class DuelingDQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DuelingDQN, self).__init__()

        self.noisy_layers = [
            NoisyFactorizedLinear(256, 64),
            NoisyFactorizedLinear(64, n_actions)
        ]

        self.mlp = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.fc_adv = nn.Sequential(
            #nn.Linear(256, 64),
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
            #nn.Linear(64, n_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def adv_val(self, x):
        x = self.mlp(x)
        return self.fc_adv(x), self.fc_val(x)
    
    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(axis=1, keepdim=True))

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


class NoisyDQN(nn.Module):
    def __init__(self, obs_size, action_size):
        super(NoisyDQN, self).__init__()

        self.noisy_layers = [
            NoisyFactorizedLinear(128, 64),
            NoisyFactorizedLinear(64, action_size)
        ]
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def forward(self, x):
        return self.net(x)
    
    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]
    

class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features,
                 sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(1, in_features)
        self.register_buffer("epsilon_input", z1)
        z2 = torch.zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * \
                         torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        v = self.weight + self.sigma_weight * noise_v
        return F.linear(input, v, bias)


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size:int):
    buffer.populate(initial)

    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)

def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]

    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)

        is_done = exp.last_state is None
        dones.append(is_done)
        if is_done:
            lstate = state 
        else:
            lstate = np.array(exp.last_state)
        
        last_states.append(lstate)

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_prio(batch, batch_weights, net, tgt_net,
              gamma, double=False, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)

    #como la red devuelve para cada estado el valor de todas las acciones, solo tomamos el valor de las acciones que se tomaron realmente
    state_actions_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        #si se usa double q learning, se toman los valores en el estado proximo de la Q_net tomando la accion mejor para la Q_tgt_net de esta forma se quita el maximization bias
        #Q_net(best_action) = Q_net(argmax Q_tgt_net(action))
        if double:
            next_state_acts = net(next_states_v).max(1)[1].unsqueeze(-1)
            next_state_values = tgt_net(next_states_v).gather(1, next_state_acts).squeeze(-1)
        else:
            #se toma para cada experiencia del batch la inferencia y luego su máximo por cada ejemplo para calcular el loss de golpe
            next_state_values = tgt_net(next_states_v).max(1)[0]
        #los valores de los estados finales son cero
        next_state_values[done_mask] = 0.0 

    #se deja fuera del grafo de gradientes
    next_state_values = next_state_values.detach()
    target = rewards_v + gamma*next_state_values
    loss_v = (state_actions_values - target) ** 2
    loss_v_weighted = loss_v * batch_weights_v

    loss = loss_v_weighted.mean()
    new_prios = (loss_v_weighted + 1e-5).data.cpu().numpy()

    return loss, new_prios


class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector, params: SimpleNamespace):
        self.selector = selector
        self.params = params 
    
    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx/self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)
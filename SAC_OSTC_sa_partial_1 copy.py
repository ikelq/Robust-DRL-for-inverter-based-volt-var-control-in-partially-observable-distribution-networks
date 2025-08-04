# the ieee 33 bus system has been stalled with 3 IB-ERs
# change the position of those IB-ER to show a clear simulation results   
# generating the loading data
# run 118 with penality cofficient 50 for voltage violation 
# correct the state, include the PQ of slack bus for all environment
# not use sam optimizator

# add reactive power into state

import copy
import random
from typing import Dict, List, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandapower as pp
from pandas.core.frame import DataFrame
from torch.distributions import Normal
import pandas as pd
import argparse
# from sam import SAM
import datetime

from IPython.display import clear_output

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def Relu(x: np.ndarray):
    return np.maximum(0, x)

def huber(x, k=100.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, size: int, batch_size: int = 32):
        """Initializate."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size,2], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size-decay, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class GaussianNoise:
    """Gaussian Noise.
    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            log_std_min: float = -20,
            log_std_max: float = 2,
            n_layer = [512, 512],
    ):
        """Initialize."""
        super(Actor, self).__init__()

        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # set the hidden layers
        self.hidden1 = nn.Linear(in_dim, n_layer[0])
        self.hidden2 = nn.Linear(n_layer[0], n_layer[1])
        self.hidden1e = nn.Linear(in_dim, n_layer[0])
        self.hidden2e = nn.Linear(n_layer[0], n_layer[1])
        # self.hidden3 = nn.Linear(n_layer[1], n_layer[2])

        # set log_std layer
        self.log_std_layer = nn.Linear(n_layer[1], out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(n_layer[1], out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        xe = F.relu(self.hidden1e(state))
        xe = F.relu(self.hidden2e(xe))
        # x = F.relu(self.hidden3(x))

        # get mean
        mu = self.mu_layer(x).tanh()

        # get std
        log_std = self.log_std_layer(xe).tanh()
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)

        # sample actions
        dist = Normal(mu, std)
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, mu.tanh()

class CriticQ(nn.Module):
    def __init__(self,
                 in_dim: int,
                 n_layer = [512, 512]):
        """Initialize."""
        super(CriticQ, self).__init__()

        self.hidden1 = nn.Linear(in_dim, n_layer[0])
        self.hidden2 = nn.Linear(n_layer[0], n_layer[1])
        # self.hidden3 = nn.Linear(n_layer[1], n_layer[2])
        self.out = nn.Linear(n_layer[1], 1)
        self.out = init_layer_uniform(self.out)

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        value = self.out(x)

        return value

class SACAgent:

    def __init__(
            self,
            env,
            memory_size: int,
            batch_size: int,
            gamma: float = 0.9,
            tau: float = 5e-3,
            initial_random_steps: int = 1e4,
            policy_update_freq: int = 1,
            seed: int = 777,
            env_name=33,
            partial=0,
    ):
        """Initialize."""
        self.partial = partial
        obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.env = env
        self.memory = ReplayBuffer(obs_dim, self.action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq
        self.seed = seed
        self.env.name = env_name

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)


        # automatic entropy tuning
        self.target_entropy = -np.prod((self.action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # actor
        self.actor = Actor(obs_dim, self.action_dim).to(self.device)

        # q function
        self.qf_1 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target1 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target1.load_state_dict(self.qf_1.state_dict())

        self.qf_2 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target2 = CriticQ(obs_dim + self.action_dim).to(self.device)
        self.qf_target2.load_state_dict(self.qf_2.state_dict())

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=3e-4)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=3e-4)

        # self.qf_1_optimizer = SAM(self.qf_1.parameters(), torch.optim.Adam, lr=3e-4, adaptive=True)
        # self.qf_2_optimizer = SAM(self.qf_2.parameters(), torch.optim.Adam, lr=3e-4, adaptive=True)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

        self.actor_loss = torch.zeros(1)

        self.state = self.env.reset()



    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            # selected_action = self.env.action_space.sample()
            selected_action = 0.3*np.random.randn(self.action_dim)
            selected_actiont = selected_action
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            )[0].detach().cpu().numpy()
            selected_actiont = self.actor(torch.FloatTensor(state).to(self.device))[2].detach().cpu().numpy()

        if self.is_test:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            )[2].detach().cpu().numpy()

        # self.transition = [state, selected_action]

        return selected_action, selected_actiont


    def step_model(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.env.step_model(action)
        return next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state


    
    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"]).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        # next_action, log_prob, _ = self.actor(next_state)
        action_p, log_prob, _ = self.actor(state)

        # train alpha (dual problem)
        alpha_loss = (
                -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        # q function loss
        mask = 1 - done
        q_1_pred = self.qf_1(state, action)
        q_2_pred = self.qf_2(state, action)
        # v_target = self.vf_target(next_state)

        # q_pred = torch.min(self.qf_1(next_state, next_action), self.qf_2(next_state, next_action))

        q1_target = reward[:, 0].reshape(-1, 1)
        q2_target = penalty_cofficient*reward[:, 1].reshape(-1, 1)

        qf_1_loss = F.mse_loss(q_1_pred, q1_target.detach())
        if qr:
            # errors = q2_target.detach() - q_2_pred
            # qf_2_loss = torch.max((0.05 - 1) * errors, 0.05 * errors).mean()

            diff = q2_target.detach() - q_2_pred
            # loss = huber(diff) * (0.05 - (diff.detach() < 0).float()).abs()
            loss = diff.pow(2) * (0.2 - (diff.detach() < 0).float()).abs()
            #  x.pow(2)
            qf_2_loss = loss.mean()

        else:
            qf_2_loss = F.mse_loss(q_2_pred, q2_target.detach())

        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        # qf_1_loss.backward()
        # self.qf_1_optimizer.first_step(zero_grad=True)
        # F.mse_loss(self.qf_1(state, action), q1_target.detach()).backward()
        # self.qf_1_optimizer.second_step(zero_grad=True)

        # qf_2_loss.backward()
        # self.qf_2_optimizer.first_step(zero_grad=True)
        # F.mse_loss(self.qf_2(state, action), q2_target.detach()).backward()
        # self.qf_2_optimizer.second_step(zero_grad=True)

        qf_loss = qf_1_loss + qf_2_loss

        if self.total_step % self.policy_update_freq == 0:
            # actor loss
            # action_p, log_prob, _ = self.actor(state)
            advantage = self.qf_1(state, action_p) + self.qf_2(state, action_p)
            actor_loss = (alpha * log_prob - advantage).mean()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update (vf)
            # self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        # train Q functions

        return actor_loss.detach().cpu().numpy(), qf_loss.detach().cpu().numpy(),  alpha_loss.detach().cpu().numpy()

    def train(self, num_frames: int, plotting_interval: int = 400):
        """Train the agent."""
        self.is_test = False
        actor_losses, qf_losses,  alpha_losses = [], [], []
        decision_times, train_times = [], []
        scores = []
        violation_sum_s = []
        violation_sum_M_s = []
        violation_sum_N_s = []
        grid_loss_sum_s = []
        score = 0
        violation_sum = 0
        violation_sum_M = 0
        violation_sum_N = 0
        grid_loss_sum = 0

        scorest = []
        violation_sum_st = []
        violation_sum_M_st = []
        violation_sum_N_st = []
        grid_loss_sum_st = []
        scoret = 0
        violation_sumt = 0
        violation_sum_Mt = 0
        violation_sum_Nt = 0
        grid_loss_sumt = 0

        real_voltage_sum_M_st = []
        real_voltage_sum_N_st = []
        real_voltage_sum_Mt = 0
        real_voltage_sum_Nt = 0

        reward_5 = 0

        for self.total_step in range(1, num_frames + 1):

            start_time = datetime.datetime.now()
            action, action0 = self.select_action(self.state)
            end_time = datetime.datetime.now()
            decision_time = (end_time - start_time).total_seconds()


            next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.step_model(action)
            self.env.step_n = self.env.step_n - 1
            next_state1, reward1, done1, violation1, violation_M1, violation_N1, voltage_M1, voltage_N1, grid_loss1, new_state1 = self.step_model(action0)
            self.transition = [self.state, action, reward, next_state, done]
            self.memory.store(*self.transition)

            # if decay_average != 0:
            #     reward_5 += reward
            #     if self.env.step_n % decay_average == 0:
            #         self.memory.store_1(reward_5/decay_average)
            #         reward_5 = 0

            if self.env.step_n > 140000:
                self.env.step_n = 0
            # state = next_state
            self.state = new_state
            score += reward[0]+penalty_cofficient*reward[1]
            violation_sum = violation_sum + violation
            violation_sum_M = violation_sum_M + violation_M
            violation_sum_N = violation_sum_N + violation_N
            grid_loss_sum = grid_loss_sum + grid_loss

            scoret += reward1[0]+penalty_cofficient*reward1[1]
            violation_sumt = violation_sumt + violation1
            violation_sum_Mt = violation_sum_Mt + violation_M1
            violation_sum_Nt = violation_sum_Nt + violation_N1
            grid_loss_sumt = grid_loss_sumt + grid_loss1

            real_voltage_sum_Mt = real_voltage_sum_Mt + voltage_M1
            real_voltage_sum_Nt = real_voltage_sum_Nt + voltage_N1

            if self.total_step % (1000) == 0:
                # state = env.reset()
                scores.append(score)
                violation_sum_s.append(violation_sum)
                violation_sum_M_s.append(violation_sum_M)
                violation_sum_N_s.append(violation_sum_N)
                grid_loss_sum_s.append(grid_loss_sum)
                score = 0
                violation_sum = 0
                violation_sum_M = 0
                violation_sum_N = 0
                grid_loss_sum = 0

                scorest.append(scoret)
                violation_sum_st.append(violation_sumt)
                violation_sum_M_st.append(violation_sum_Mt)
                violation_sum_N_st.append(violation_sum_Nt)
                grid_loss_sum_st.append(grid_loss_sumt)
                scoret = 0
                violation_sumt = 0
                violation_sum_Mt = 0
                violation_sum_Nt = 0
                grid_loss_sumt = 0

                real_voltage_sum_M_st.append(real_voltage_sum_Mt)
                real_voltage_sum_N_st.append(real_voltage_sum_Nt)
                real_voltage_sum_Mt = 0
                real_voltage_sum_Nt = 0

            # if training is ready
            for i in range(5):
                if (
                        len(self.memory) >= self.batch_size
                        and self.total_step > self.initial_random_steps
                ):
                    losses = self.update_model()
            if 'losses' in locals():
                actor_losses.append(losses[0])
                qf_losses.append(losses[1])
                alpha_losses.append(losses[2])

            # plotting
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step,
                    scores,
                    violation_sum_s,
                    violation_sum_M_s,
                    violation_sum_N_s,
                    grid_loss_sum_s,
                    actor_losses,
                    qf_losses,
                    alpha_losses
                )


            train_end_time = datetime.datetime.now()
            train_time = (train_end_time - start_time).total_seconds()

            decision_times.append(decision_time)
            train_times.append(train_time)




        data_tr = {"scores": scores,
                   "violation_sum_s": violation_sum_s,
                   "violation_sum_M_s": violation_sum_M_s,
                   "violation_sum_N_s": violation_sum_N_s,
                   "grid_loss_sum_s": grid_loss_sum_s
                   }
        data_train = DataFrame(data_tr)
        data_train.to_csv('trainsac' + str(self.env.name) + str(self.seed) + str(self.partial) + 'tootsa'+str(qr)+str(decay)+str(decay_average)+'.csv')

        data_tr_test = {"scorest": scorest,
                        "violation_sum_st": violation_sum_st,
                        "violation_sum_M_st": violation_sum_M_st,
                        "violation_sum_N_st": violation_sum_N_st,
                        "grid_loss_sum_st": grid_loss_sum_st,
                        "real_voltage_sum_M_st": real_voltage_sum_M_st,
                        "real_voltage_sum_N_st": real_voltage_sum_N_st,
                        }
        data_train_test = DataFrame(data_tr_test)
        data_train_test.to_csv('traintestsac' + str(self.env.name) + str(self.seed) + str(self.partial) + 'tootsa'+str(qr)+str(decay)+str(decay_average)+'.csv')

        data_tr_loss = {"actor_losses": actor_losses,
                        "qf_losses": qf_losses,
                        "alpha_losses": alpha_losses
                        }
        data_train_loss = DataFrame(data_tr_loss)
        data_train_loss.to_csv('trainlosssac' + str(self.env.name) + str(self.seed) + str(self.partial) + 'tootsa'+str(qr)+str(decay)+str(decay_average)+'.csv')

        data_time = {"decision_times": decision_times,
                "train_times": train_times
                }
        data_time = DataFrame(data_time)
        data_time.to_csv('data_time' + str(self.env.name) + str(self.seed) + str(self.partial) + 'tootsa'+str(qr)+str(decay)+str(decay_average)+'.csv')

        #
        # torch.save(self.log_alpha, 'log_alpha_params' + 'sac'+'.pth')
        # torch.save(self.actor.state_dict(), 'actor_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_1.state_dict(), 'critic1_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_2.state_dict(), 'critic2_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_target1.state_dict(), 'critic_target1_p_params' + 'sac' + '.pth')
        # torch.save(self.qf_target2.state_dict(), 'critic_target2_p_params' + 'sac' + '.pth')



    def test(self, test_frams):
        """Test the agent."""
        self.is_test = True
        self.env.step_n = test_frams-1
        # state = self.env.reset()
        state = self.state

        # # initial state for real model
        action, _ = self.select_action(state)
        next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss, new_state = self.step_model(action)

        state = new_state

        device = self.device  # for shortening the following lines

        scores_t = []
        violation_sum_s_t = []
        violation_sum_M_s_t = []
        violation_sum_N_s_t = []
        grid_loss_sum_s_t = []
        score = 0
        violation_sum = 0
        violation_sum_M = 0
        violation_sum_N = 0
        grid_loss_sum = 0

        violation_s_t = []
        violations_M_t = []
        violations_N_t = []
        grid_loses_t = []
        actions= []


        while self.env.step_n < 96*390:
            action, _ = self.select_action(state)
            next_state, reward, done, violation, violation_M, violation_N, voltage_M, voltage_N, grid_loss,  new_state = self.step_model(action)

            violation_s_t.append(violation)
            violations_M_t.append(violation_M)
            violations_N_t.append(violation_N)
            grid_loses_t.append(grid_loss)
            actions.append(action)

            # state = next_state
            state = new_state
            score += reward[0]+penalty_cofficient*reward[1]
            violation_sum = violation_sum + violation
            violation_sum_M = violation_sum_M + violation_M
            violation_sum_N = violation_sum_N + violation_N
            grid_loss_sum = grid_loss_sum + grid_loss

            if self.env.step_n % 96 == 0:
                scores_t.append(score)
                violation_sum_s_t.append(violation_sum)
                violation_sum_M_s_t.append(violation_sum_M)
                violation_sum_N_s_t.append(violation_sum_N)
                grid_loss_sum_s_t.append(grid_loss_sum)
                score = 0
                violation_sum = 0
                violation_sum_M = 0
                violation_sum_N = 0
                grid_loss_sum = 0

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (151, f"scores_t", scores_t),
            (152, f"violation_sum_s_t", violation_sum_s_t),
            (153, f"violation_sum_M_s_t", violation_sum_M_s_t),
            (154, f"violation_sum_N_s_t", violation_sum_N_s_t),
            (155, "grid_loss_sum_s_t", grid_loss_sum_s_t)
        ]

        clear_output(True)
        plt.figure(figsize=(30, 6))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()

        data_te = {"scores_t": scores_t,
                   "violation_sum_s_t": violation_sum_s_t,
                   "violation_sum_M_s_t": violation_sum_M_s_t,
                   "violation_sum_N_s_t": violation_sum_N_s_t,
                   "grid_loss_sum_s_t": grid_loss_sum_s_t,
                   }

        data_test = DataFrame(data_te)
        data_test.to_csv('testsac'+str(self.env.name)+str(self.seed) +'tootsa.csv')

        data_te_step = {"violation_s_t": violation_s_t,
                   "violations_M_t": violations_M_t,
                   "violations_N_t": violations_N_t,
                   "grid_loses_t": grid_loses_t,
                   }
        data_test_step = DataFrame(data_te_step)
        data_test_step.to_csv('testsac_step' +str(self.env.name) + str(self.seed) + 'tootsa.csv')
        data_test_step_action = DataFrame(actions)
        data_test_step_action.to_csv(
            'testsac_step_action' + str(self.env.name) + str(self.seed) + 'tootsa.csv')

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        # for t_param, l_param in zip(
        #         self.vf_target.parameters(), self.vf.parameters()
        # ):
        #     t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        for t_param, l_param in zip(
                self.qf_target1.parameters(), self.qf_1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        for t_param, l_param in zip(
                self.qf_target2.parameters(), self.qf_2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            violation_sum_s: List[float],
            violation_sum_M_s: List[float],
            violation_sum_N_s: List[float],
            grid_loss_sum_s: List[float],
            actor_losses: List[float],
            qf_losses: List[float],
            alpha_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (241, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (242, f"actor_loss", actor_losses),
            (243, f"qf_loss", qf_losses),
            (244, "alpha_loss ", alpha_losses),
            (245, f"violation_sum_s {self.env.name}", violation_sum_s),
            (246, "violation_sum_M_s", violation_sum_M_s),
            (247, "violation_sum_N_s", violation_sum_N_s),
            (248, f"grid_loss_sum_s {np.mean(grid_loss_sum_s[-10:])}", grid_loss_sum_s),
        ]

        clear_output(True)
        plt.figure(figsize=(20, 10))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--partial", default=11, type=int) 
    parser.add_argument("--qr", default=0, type=int) 
    parser.add_argument("--env_name", default=33, type=int) 
    args = parser.parse_args()

    load_pu = np.load('load96.npy')
    gene_pu = np.load('gen96.npy')

    # parameters
    num_frames = 100000
    # test_frames = 96*360
    memory_size = 100000
    batch_size = 128
    initial_random_steps = 10000
    decay = 0    # decay time for sampling data form data buffer
    decay_average = 0   # eahc dacay_average_reward data have the same reward

    # partial = 1 : partial power loss; partial = 2 : partial power loss and voltage violation;
    # partial = 3 : partial state power loss voltage violation   # partial = 4 : partial state #partial = 5 partial state, reduce loss
    partial = args.partial
    # qr = args.qr
    env_name = args.env_name

    # if partial == 0:
    #     import Env as Env
    # if partial == 1:
    #     import Env_partial_power_loss as Env
    if partial == 2:
        import Env_partial_voltage as Env
        # torch.cuda.set_device(2)
    if partial == 3:
        import Env_partial_reward as Env
        # torch.cuda.set_device(2)
    # if partial == 4:
    #     import Env_partial_state as Env
    # if partial == 5:
    #     import Env_partial_state_power_loss as Env
    if partial == 6:
        import Env_partial_state_voltage as Env
        # torch.cuda.set_device(3)
    if partial == 7:
        import Env_partial_state_reward as Env
        # torch.cuda.set_device(3)
    # if partial == 8:
    #     import Env_li as Env


    if partial == 11:
        import Env_partial_voltage_li as Env
        # torch.cuda.set_device(4)

    if partial == 12:
        import Env_partial_reward_li as Env
        # torch.cuda.set_device(4)
    # if partial == 13:
    #     import Env_partial_state_li as Env
    if partial == 15:
        import Env_partial_state_voltage_li as Env
        # torch.cuda.set_device(5)
    if partial == 16:
        import Env_partial_state_reward_li as Env
        # torch.cuda.set_device(5)

    if partial == 17:
        import Env_partial_voltage_without_load as Env
        # torch.cuda.set_device(6)
    if partial == 18:
        import Env_partial_reward_without_load as Env
        # torch.cuda.set_device(6)
    if partial == 19:
        import Env_partial_state_voltage_without_load as Env
        # torch.cuda.set_device(1)
    if partial == 20:
        import Env_partial_state_reward_without_load as Env
        # torch.cuda.set_device(1)

    print(partial)
    
    for seed in [777]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        for qr in [0,1]:
            # for env_name in [33,69,118]:
            print(env_name)

            if env_name == 69:
                # ieee_model = pc.from_mpc('case69.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
                id_iber = [5, 22, 44, 63]
                id_svc = [52]  #, 33
                line_f_bus = [2,7,11,36]
                penalty_cofficient = 50
                id_svc_capacity = 2
                iber_re_capacity = 3
            if env_name ==33:
                # ieee_model = pc.from_mpc('case33_bw.mat', f_hz=50, casename_mpc_file='mpc', validate_conversion=False)
                id_iber = [16, 31]
                id_svc = [23]
                line_f_bus = [2,5,10]
                penalty_cofficient = 50
                id_svc_capacity = 2
                iber_re_capacity = 3
            if env_name == 118:
                # id_iber = [33, 50, 53, 69, 76, 97, 106, 111]
                # id_svc = [44, 104]
                id_iber = [33, 44, 50, 53, 76, 97, 106, 111]
                id_svc = [69, 84]
                line_f_bus = [1,10,28,29,64,78,99]
                penalty_cofficient = 50
                id_svc_capacity = 2
                iber_re_capacity = 3
            #
            
            env = Env.grid_case(env_name, load_pu, gene_pu, id_iber, id_svc, line_f_bus, iber_re_capacity,
                id_svc_capacity)
            
            # partial = 1 : partial power loss; partial = 2 : partial power loss and voltage violation;
            # partial = 3 : partial state power loss voltage violation   # partial = 4 : partial state #partial = 5 partial state, reduce loss
            agent = SACAgent(
                env,
                memory_size,
                batch_size,
                initial_random_steps=initial_random_steps,
                seed = seed,
                env_name = env_name,
                partial= partial
            )

            agent.train(num_frames, plotting_interval=4000000)
        # agent.test(test_frames)

#  python SAC_OSTC_sa_partial_1.py --partial 0 &
#  python SAC_OSTC_sa_partial_1.py --partial 2 &
#  python SAC_OSTC_sa_partial_1.py --partial 4 &
#  python SAC_OSTC_sa_partial_1.py --partial 5 &
#  python SAC_OSTC_sa_partial_1.py --partial 6 
    # f = open('agent.pkl', 'wb')
    # pickle.dump(agent, f)
    # f.close()
    #  save buffer only eliminating voltage violation
    # f = open('agentmemory33rand.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()
    #
    # f = open('agentmemory69rand.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()
    # f = open('agentmemory'+str(env_name)+'randnoise'+str(noise)+'.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()

    # f = open('agentmemory69expertnosie.pkl', 'wb')
    # pickle.dump(agent.memory, f)
    # f.close()
    #
    # fl = open('agentmemory.pkl', 'rb')
    # memory1 = pickle.load(fl)
    # fl.close()


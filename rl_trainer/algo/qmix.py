import os
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from common import device

from replay_buffer import QMIXReplayBuffer


HYPPER_HIDDEN_SIZE=256
QMIX_HIDDEN_SIZE = 256
DRQN_HIDDEN_SIZE = 256
HIDDEN_SIZE = 256



class QMIX:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        # self.conf = conf
        # self.device = self.conf.device
        # self.n_actions = self.conf.n_actions
        # self.n_agents = self.conf.n_agents
        # self.state_shape = self.conf.state_shape
        # self.obs_shape = self.conf.obs_shape
        # input_shape = self.obs_shape
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation
        self.update_interval = args.update_interval

        # args for qmix
        self.episode_length = args.episode_length
        

        # print(self.device, self.n_actions, self.n_agents, self.state_shape, self.obs_shape, input_shape)

        # NET
        self.eval_drqn_net = DRQN(obs_dim, act_dim, num_agent, args).to(self.device)
        self.target_drqn_net = DRQN(obs_dim, act_dim, num_agent, args).to(self.device)

        self.eval_qmix_net = QMIXNet(obs_dim, act_dim, num_agent, args).to(self.device)
        self.target_qmix_net = QMIXNet(obs_dim, act_dim, num_agent, args).to(self.device)


        # self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_drqn_net.parameters())
        self.drqn_optimizer = torch.optim.Adam(self.eval_drqn_net.parameters(), lr=self.a_lr)
        self.qmix_optimizer = torch.optim.Adam(self.eval_qmix_net.parameters(), lr=self.c_lr)

        # data: [batch_size, episode_length, n_agents, _]


        self.replay_buffer = QMIXReplayBuffer(args.buffer_size, args.batch_size, args.episode_length)


    def init_hidden(self, batch_size):
        self.eval_hidden = torch.zeros((batch_size, self.num_agent, DRQN_HIDDEN_SIZE)).to(device)
        self.target_hidden = torch.zeros((batch_size, self.num_agent, DRQN_HIDDEN_SIZE)).to(device)

    
    def choose_action(self, obs, evaluation=False):

        obs = torch.Tensor(obs).to(self.device)
        # print(obs.shape)
        # self.eval_hidden = self.eval_hidden[0].to(self.device)
        q_value, self.eval_hidden = self.eval_drqn_net(obs, self.eval_hidden)

        # print(obs, q_value)

        p = np.random.random()
        if p > self.eps or evaluation:
            # action = np.zeros((self.num_agent, self.act_dim))
            action = torch.argmax(q_value, dim=1).cpu().numpy()
            # print(q_value.shape, torch.max(q_value, dim=1, keepdim=True))
            # for i in range(self.num_agent):
            #     action[i][max_action[i]] = 1
        else:
            action = self.random_action()
        
        self.eps *= self.decay_speed
        return action


    def random_action(self):
        return np.random.choice(np.array([0, 1, 2, 3]), size=(3))
        # if self.output_activation == 'tanh':
        #     return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        # return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))

    
    def update(self, cur_step):

        self.init_hidden(self.batch_size)

        # here state is obs
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, all_state_batch, all_next_state_batch = self.replay_buffer.get_batches()

        # (batch_size, max_episode_len, n_agents, _)
        # q_evals, q_targets = self.get_q_values(state_batch, self.eval_hidden.to(device))

        state_batch = torch.Tensor(state_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).to(self.device)
        done_batch = torch.Tensor(done_batch).to(self.device)

        action_batch = action_batch.to(torch.int64).unsqueeze(-1)
        # print(action_batch)

        q_evals = []
        q_targets = []

        for idx in range(self.episode_length):
            # input: (batch_size, n_agents, obs_dim)
            # output: (batch_size, n_agents, n_actions)
            q_eval, self.eval_hidden = self.eval_drqn_net(state_batch[:, idx].reshape(-1, self.obs_dim), self.eval_hidden)
            q_target, self.target_hidden = self.target_drqn_net(next_state_batch[:, idx].reshape(-1, self.obs_dim), self.target_hidden)
            q_evals.append(q_eval.reshape(-1, self.num_agent, self.act_dim))
            q_targets.append(q_target.reshape(-1, self.num_agent, self.act_dim))
            

        # (batch_size, episode_len, n_agents, n_actions)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)


        # (batch_size, episode_len, n_agents)
        # print(q_evals.shape, action_batch.shape)
        q_evals = torch.gather(q_evals, dim=3, index=action_batch).squeeze(3)
        q_targets = torch.max(q_targets, dim=-1)[0]

        # print(q_evals.shape, q_targets.shape)

        # q_evals = torch.max(q_evals, dim=-1)[0]
        # 
        
        # print(all_state_batch.shape, all_next_state_batch.shape)
        qmix_eval = self.eval_qmix_net(q_evals, all_state_batch)  # (batch_size, n_agents, 1)
        qmix_target = self.target_qmix_net(q_targets, all_next_state_batch)

        

        expected_q = reward_batch + self.gamma * (1-done_batch) * qmix_target

        print(done_batch.detach().cpu().numpy().mean(), reward_batch.detach().cpu().numpy().mean(), qmix_eval.detach().cpu().numpy().mean(), expected_q.detach().cpu().numpy().mean())

        loss = torch.mean(F.mse_loss(qmix_eval, expected_q.detach()))

        print("loss", loss.item())

        # self.optimizer.zero_grad()
        self.drqn_optimizer.zero_grad()
        self.qmix_optimizer.zero_grad()
        loss.backward()
        self.drqn_optimizer.step()
        self.qmix_optimizer.step()

        # update target network
        if cur_step % self.update_interval == 0:
            print('update model')
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        return torch.mean(reward_batch).item(), loss.item()


    def save_model(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "drqn_" + str(episode) + ".pth")
        torch.save(self.eval_drqn_net.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "qmix_" + str(episode) + ".pth")
        torch.save(self.eval_qmix_net.state_dict(), model_critic_path)



class DRQN(nn.Module):

    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='softmax'):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, DRQN_HIDDEN_SIZE)
        self.rnn = nn.GRUCell(DRQN_HIDDEN_SIZE, DRQN_HIDDEN_SIZE)
        self.fc2 = nn.Linear(DRQN_HIDDEN_SIZE, act_dim)

    
    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, DRQN_HIDDEN_SIZE)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    

class QMIXNet(nn.Module):

    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='softmax'):
        super().__init__()

        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = obs_dim * num_agents
        self.num_agents = num_agents

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, HYPPER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HYPPER_HIDDEN_SIZE, num_agents * QMIX_HIDDEN_SIZE)
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, HYPPER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HYPPER_HIDDEN_SIZE, QMIX_HIDDEN_SIZE)
        )

        self.hyper_b1 = nn.Linear(self.state_dim, QMIX_HIDDEN_SIZE)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, HYPPER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HYPPER_HIDDEN_SIZE, 1)
        )

    def forward(self, q_values, states):
        
        batch_size = q_values.shape[0]
        
        # (batch_size, 1, num_agents)
        q_values = q_values.view(-1, 1, self.num_agents)

        # (batch_size, obs_dim)
        # print(q_values, states)
        states = states.reshape(-1, self.state_dim)
        states = torch.Tensor(states).to(device)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.num_agents, QMIX_HIDDEN_SIZE)
        b1 = b1.view(-1, 1, QMIX_HIDDEN_SIZE)

        # (batch_size, 1, self.num_agents) * (batch_size, self.num_agents, QMIX_HIDDEN_SIZE) -> (batch_size, 1, QMIX_HIDDEN_SIZE)
        # print(q_values.shape, w1.shape, b1.shape, torch.bmm(q_values, w1).shape)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, QMIX_HIDDEN_SIZE, 1)
        b2 = b2.view(-1, 1, 1)

        # (batch_size, 1, QMIX_HIDDEN_SIZE) * (batch_size, QMIX_HIDDEN_SIZE, 1) -> (batch_size, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(batch_size, -1, 1)

        return q_total
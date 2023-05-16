import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
import random
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device, mlp, greedy_main
# from algo.network import Actor, Critic


HIDDEN_SIZE = 256


class MAPPO:

    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps_greedy = args.epsilon_greedy
        self.eps_clip = args.epsilon_clip
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation
        self.episode_length = args.episode_length
        self.eps = 0

        # Initialise actor network and critic network with ξ and θ
        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        # Initialise target network and critic network with ξ' ← ξ and θ' ← θ
        # self.actor_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        # self.critic_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        # hard_update(self.actor, self.actor_target)
        # hard_update(self.critic, self.critic_target)

        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.c_loss = None
        self.a_loss = None

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):

        p = np.random.random()
        if p > self.eps_greedy or evaluation:
            obs = torch.Tensor([obs]).to(self.device)
            action = self.actor(obs).cpu().detach().numpy()[0]
            action_dict = torch.distributions.Categorical(torch.Tensor(action))
            action = action_dict.sample().numpy()
            # print(action.shape)
        else:
            action = np.random.choice(np.array([0, 1, 2, 3]), size=(3))

        self.eps_greedy *= self.decay_speed
        return action
    
    def greedy_init(self, obs, state, eps):
        
        if eps < 200:
            action = np.array(greedy_main(state)).squeeze(1)
        else:
            obs = torch.Tensor([obs]).to(self.device)
            action = self.actor(obs).cpu().detach().numpy()[0]
            action_dict = torch.distributions.Categorical(torch.Tensor(action))
            action = action_dict.sample().numpy()

        # self.eps +=1
        # print(self.eps)

        return action

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))
    
    def compute_advantage(self, td_delta):
        td_delta = td_delta.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
            # print(advantage.shape)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float).to(self.device)

    def update(self, batch):

        # Sample a greedy_min mini-batch of M transitions from R
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, global_state_batch = batch

        # print(state_batch.shape) # (batch_size, num_agent, obs_dim)

        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        # action_batch = torch.argmax(action_batch, dim=-1, keepdim=True).to(self.device)
        # print(action_batch.shape)
        reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)

        action_batch = action_batch.to(torch.int64)


        # print(reward_batch.shape, done_batch.shape)
        with torch.no_grad():
            td_target = reward_batch + self.gamma * self.critic(next_state_batch) * (1 - done_batch)
            td_delta = td_target - self.critic(state_batch)
            advantage = self.compute_advantage(td_delta)
            old_prob = torch.clamp(self.actor(state_batch).gather(2, action_batch), 1e-10, 1.0)
            old_log_prob = torch.log(old_prob).detach()
        
        actor_losses = []
        critic_losses = []

        for _ in range(16):
            index = np.random.choice(np.arange(self.batch_size), size=512, replace=False)
            mini_state_batch = state_batch[index]
            mini_action_batch = action_batch[index]
            mini_old_log_prob = old_log_prob[index]
            mini_advantage = advantage[index]
            mini_td_target = td_target[index]
            mini_reward_batch = reward_batch[index]

            # prob = torch.clamp(self.actor(state_batch).gather(1, action_batch), 1e-10, 1.0)
            # log_prob = torch.log(prob)
            # ratio = torch.exp(log_prob - old_log_prob)
            # surr1 = ratio * advantage
            # surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
            # actor_loss = torch.mean(-torch.min(surr1, surr2))
            # critic_loss = torch.mean(
            #     F.mse_loss(self.critic(state_batch), td_target.detach()))
            # print(mini_action_batch)
            prob = torch.clamp(self.actor(mini_state_batch).gather(2, mini_action_batch), 1e-10, 1.0)
            log_prob = torch.log(prob)
            ratio = torch.exp(log_prob - mini_old_log_prob)
            # print(torch.mean(ratio).item())
            surr1 = ratio * mini_advantage
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mini_advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            critic_loss = torch.mean(
                F.mse_loss(self.critic(mini_state_batch), mini_td_target.detach()))
            # print("2")
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # print(actor_loss.item(), critic_loss.item(), 
            #       torch.max(mini_td_target).item(), torch.mean(mini_td_target).item(), 
            #       torch.max(self.critic(mini_state_batch)).item(), torch.mean(self.critic(mini_state_batch)).item(), 
            #       torch.max(mini_reward_batch).item(), torch.mean(mini_reward_batch).item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        return np.mean(np.array(actor_losses)), np.mean(np.array(critic_losses)), torch.mean(mini_reward_batch).item()


    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)




class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE]

        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)

        out = self.post_dense(out)
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim*3, HIDDEN_SIZE]

        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, num_agents]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post)

    def forward(self, obs_batch):
        # out = torch.cat((obs_batch, action_batch), dim=-1)
        obs_batch = obs_batch.reshape(-1, self.obs_dim*self.num_agents)
        out = self.prev_dense(obs_batch)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out = self.post_dense(out)
        out = out.reshape(-1, self.num_agents, 1)
        return out



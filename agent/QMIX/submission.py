import os
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


HYPPER_HIDDEN_SIZE=256
QMIX_HIDDEN_SIZE = 256
DRQN_HIDDEN_SIZE = 256

device =  torch.device("cpu")

from typing import Union
Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': torch.nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)


def get_surrounding(state, width, height, x, y, agent, info):

    state = state.copy()
    state[state>=2] = 2 

    # 0：空地 1：豆子 2: 身子 3：队友的头 4：敌人的头

    indexs_min = 3 if agent > 2 else 0

    our_index = [indexs_min, indexs_min + 1, indexs_min + 2]

    for l in [2, 3, 4, 5, 6, 7]:
        if agent + 2 == l:
            state[info[l][0][0]][info[l][0][1]] = 2
        elif agent in our_index:
            state[info[l][0][0]][info[l][0][1]] = 3
        else:
            state[info[l][0][0]][info[l][0][1]] = 4

    surrounding = np.zeros((24,5))

    surrounding[0][state[(y - 2) % height][x]] = 1  # upup
    surrounding[1][state[(y + 2) % height][x]] = 1
    surrounding[2][state[y][(x - 2) % width]] = 1
    surrounding[3][state[y][(x + 2) % width]] = 1
    surrounding[4][state[(y - 1) % height][(x - 1) % width]] = 1
    surrounding[5][state[(y - 1) % height][x]] = 1
    surrounding[6][state[(y - 1) % height][(x + 1) % width]] = 1
    surrounding[7][state[y][(x - 1) % width]] = 1
    surrounding[8][state[y][(x + 1) % width]] = 1
    surrounding[9][state[(y + 1) % height][(x - 1) % width]] = 1
    surrounding[10][state[(y + 1) % height][x]] = 1
    surrounding[11][state[(y + 1) % height][(x + 1) % width]] = 1
    surrounding[12][state[(y - 3) % height][x]] = 1
    surrounding[13][state[(y + 3) % height][x]] = 1
    surrounding[14][state[y][(x - 3) % width]] = 1
    surrounding[15][state[y][(x + 3) % width]] = 1
    surrounding[16][state[(y - 2) % height][(x - 1) % width]] = 1
    surrounding[17][state[(y - 2) % height][(x + 1) % width]] = 1
    surrounding[18][state[(y + 2) % height][(x - 1) % width]] = 1
    surrounding[19][state[(y + 2) % height][(x + 1) % width]] = 1
    surrounding[20][state[(y - 1) % height][(x - 2) % width]] = 1
    surrounding[21][state[(y - 1) % height][(x + 2) % width]] = 1
    surrounding[22][state[(y + 1) % height][(x - 2) % width]] = 1
    surrounding[23][state[(y + 1) % height][(x + 2) % width]] = 1

    surrounding = list(surrounding.flatten().tolist())

    return surrounding



def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations(state, agents_index, obs_dim, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state_ = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding(state_, width, height, head_x, head_y, element, state_copy)
        observations[i][2:122] = head_surrounding[:]

        # beans positions
        # observations[i][122:132] = beans_position[:]

        # other snake positions
        # snake_heads = np.array([snake[0] for snake in snakes_position])
        # snake_heads = np.delete(snake_heads, i, 0)
        # observations[i][132:] = snake_heads.flatten()[:]
    return observations


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



class RLAgent(object):
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.output_activation = 'softmax'
        self.actor = DRQN(obs_dim, act_dim, num_agent, self.output_activation).to(self.device)
        self.hidden = None
        self.init_hidden(1)

    def choose_action(self, obs):

        obs = torch.Tensor(obs).to(self.device)
        # print(obs.shape)
        # self.eval_hidden = self.eval_hidden[0].to(self.device)
        q_value, self.hidden = self.actor(obs, self.hidden)

        # action = np.zeros((self.num_agent, self.act_dim))
        action = torch.argmax(q_value, dim=1).cpu().numpy()
        return action

    def select_action_to_env(self, obs, ctrl_index):
        actions = self.choose_action(obs)
        # print(logits.shape)
        # actions = logits2action(logits)
        # print(actions)
        action_to_env = to_joint_action(actions, ctrl_index)
        return action_to_env

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def init_hidden(self, batch_size):
        self.eval_hidden = torch.zeros((batch_size, self.num_agent, DRQN_HIDDEN_SIZE)).to(device)
        self.target_hidden = torch.zeros((batch_size, self.num_agent, DRQN_HIDDEN_SIZE)).to(device)


def to_joint_action(action, ctrl_index):
    joint_action_ = []
    action_a = action[ctrl_index]
    each = [0] * 4
    each[action_a] = 1
    joint_action_.append(each)
    return joint_action_


def logits2action(logits):
    logits = torch.Tensor(logits).to(device)
    actions = np.array([Categorical(out).sample().item() for out in logits])
    return np.array(actions)



agent = RLAgent(122, 4, 3)
actor_net = os.path.dirname(os.path.abspath(__file__)) + "/drqn_30000.pth"
agent.load_model(actor_net)
agent.hidden = torch.zeros(3, DRQN_HIDDEN_SIZE).to(device)


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 122
    obs = observation_list.copy()
    board_width = obs['board_width']
    board_height = obs['board_height']
    o_index = obs['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
    observation = get_observations(obs, indexs, obs_dim, height=board_height, width=board_width)
    actions = agent.select_action_to_env(observation, indexs.index(o_index-2))
    return actions

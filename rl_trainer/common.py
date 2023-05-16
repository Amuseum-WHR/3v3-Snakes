import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml
from greedy_utils import greedy_main

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
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


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding2(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations2(state, agents_index, obs_dim, height, width):
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
    state = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding2(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, i, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations


def get_grid_observations(state, agents_index, obs_dim, height, width):
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
    # print(state_)
    state_ = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    for i, element in enumerate(agents_index):
        # # self head position
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]
        observations[i][:] = grid_observation(state_, width, height, head_x, head_y, element, state_copy)

    return observations


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
    # print(state_)
    state_ = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        # dim0: 10, dim1:20
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        # height = 10, width = 20
        head_surrounding = get_surrounding(state_, width, height, head_x, head_y, element, state_copy)
        observations[i][2:122] = head_surrounding[:]

        # beans positions
        # observations[i][122:132] = beans_position[:]

        # # other snake positions
        # snake_heads = np.array([snake[0] for snake in snakes_position])
        # snake_heads = np.delete(snake_heads, i, 0)
        # observations[i][132:] = snake_heads.flatten()[:]
    return observations

def manhattan(x,y,bean_x,bean_y,width,height):
    if abs(x-bean_x)>abs(width - abs(x - bean_x)):
        d_x=abs(width - abs(x - bean_x))
        if x>bean_x:
            ind_x=bean_x+width
        else:
            ind_x=bean_x-width
    else:
        d_x=abs(x-bean_x)
        ind_x=bean_x

    if abs(y-bean_y)>abs(height - abs(y - bean_y)):
        d_y=abs(height - abs(y - bean_y))
        if y>bean_y:
            ind_y=bean_y+height
        else:
            ind_y=bean_y-height
    else:
        d_y=abs(y-bean_y)
        ind_y=bean_y
    return d_x+d_y,ind_x,ind_y




def get_reward(info, snake_index, reward, score):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if score == 1:
            step_reward[i] += 50
        elif score == 2:
            step_reward[i] -= 20
        elif score == 3:
            step_reward[i] += 5
        elif score == 4:
            step_reward[i] -= 2

        if reward[i] > 0:
            step_reward[i] += 20
        else:
            self_head = np.array(snake_heads[i])
            # dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            dists = [manhattan(self_head[0], self_head[1], other_head[0], other_head[1], 20, 10)[0] for other_head in beans_position]
            step_reward[i] -= min(dists)
            if reward[i] < 0:
                step_reward[i] += reward[i] * 10

    return step_reward



def get_dense_reward(info, snake_index, reward, pre_beans, score):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    pre_beans_position = np.array(pre_beans, dtype=object)
    # print(beans_position.shape, pre_beans_position.shape, "hello")
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if score == 1:
            step_reward[i] += 50
        elif score == 2:
            step_reward[i] -= 20
        elif score == 3:
            step_reward[i] += 5
        elif score == 4:
            step_reward[i] -= 2
        # step_reward += len(snakes_position[i])-3
        if reward[i] > 0:
            step_reward[i] += 20
        else:
            self_head = np.array(snake_heads[i])
            # dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            dists = [manhattan(self_head[0], self_head[1], other_head[0], other_head[1], 10, 20)[0] for other_head in beans_position]
            pre_dists = [manhattan(self_head[0], self_head[1], other_head[0], other_head[1], 10, 20)[0] for other_head in pre_beans_position]
            step_reward[i] += (min(np.array(pre_dists))-min(np.array(dists))) * 5
            
            if reward[i] < 0:
                step_reward[i] += reward[i] * 10
    
    # enemy punish and global reward
    enemy_snake_index = [3, 4, 5]
    enemy_reward = 0
    for i in enemy_snake_index:
        if reward[i] > 0:
            enemy_reward += 20
        elif reward[i] < 0:
            enemy_reward += reward[i] * 10
    delta_reward = sum([len(snake) for snake in snakes_position[:3]]) - sum([len(snake) for snake in snakes_position[3:]])
    step_reward = step_reward + 0.1 * delta_reward - 0.1 * enemy_reward


    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def action_random(act_dim, logits):
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = logits[:]
    return actions


def logits_greedy(state, logits, height, width):
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
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list


def logits_more_greedy(state, logits, height, width):
    greedy_action = greedy_main(state)

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list


def action_more_greedy(state, logits, height, width):

    greedy_action = [i[0] for i in greedy_main(state)]
    # print(greedy_main(state))

    logits_action = logits
    # print(greedy_action)

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list


def action_greedy(state, logits, height, width):
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
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    # logits = torch.Tensor(logits).to(device)
    # logits_action = np.array([Categorical(out).sample().item() for out in logits])
    logits_action = logits

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])
    # print(logits_action, greedy_action)
    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list


def get_global_state(state):
    state_copy = state.copy()
    width = state_copy['board_width']
    height = state_copy['board_height']
    global_state = np.zeros((height, width))
    beans = state_copy [1]
    for i in beans:
        state[i[0], i[1]] = 1
    for l in [2, 3, 4]:
        for i in state_copy[l]:
            state[i[0], i[1]] = 2
    for i in [5, 6, 7]:
        for i in state_copy[l]:
            state[i[0], i[1]] = 3

    return global_state


def get_surrounding2(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding

def grid_observation(state, width, height, x, y, agent, info):
    state = state.copy()
    state[state>=2] = 2
    indexs_min = 3 if agent > 2 else 0

    our_index = [indexs_min, indexs_min + 1, indexs_min + 2]

    for l in [2, 3, 4, 5, 6, 7]:
        if agent + 2 == l:
            state[info[l][0][0]][info[l][0][1]] = 2
        elif agent in our_index:
            state[info[l][0][0]][info[l][0][1]] = 3
        else:
            state[info[l][0][0]][info[l][0][1]] = 4

    
    one_hot_state = np.zeros((height, width, 5))
    for i in range(5):
        one_hot_state[:,:,i] = (state == i).astype(int)

    # one_hot_state = one_hot_state.roll(-y, 0).roll(-x, 1).flatten()
    one_hot_state = np.roll(one_hot_state, (-y, -x), (0, 1)).flatten()
    return one_hot_state




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




def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()






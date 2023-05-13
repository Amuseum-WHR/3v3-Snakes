import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.ddpg import DDPG
from common import *
from log_path import *
from env.chooseenv import make


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 142
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model = DDPG(obs_dim, act_dim, ctrl_agent_num, args)

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()

        beans = env.beans_position.copy()

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        # print(state)
        # [{1: [[9, 6], [7, 12], [6, 4], [9, 5], [1, 1]], 2: [[3, 10], [3, 9], [3, 8]], 3: [[3, 15], [2, 15], [1, 15]], 4: [[5, 1], [4, 1], [3, 1]], 5: [[3, 19], [4, 19], [5, 19]], 6: [[6, 14], [7, 14], [8, 14]], 7: [[7, 19], [8, 19], [9, 19]], 
        #   'board_width': 20, 'board_height': 10, 'last_direction': None, 'controlled_snake_index': 2}, 
        # {1: [[9, 6], [7, 12], [6, 4], [9, 5], [1, 1]], 2: [[3, 10], [3, 9], [3, 8]], 3: [[3, 15], [2, 15], [1, 15]], 4: [[5, 1], [4, 1], [3, 1]], 5: [[3, 19], [4, 19], [5, 19]], 6: [[6, 14], [7, 14], [8, 14]], 7: [[7, 19], [8, 19], [9, 19]], 'board_width': 20, 'board_height': 10, 'last_direction': None, 'controlled_snake_index': 3}, {1: [[9, 6], [7, 12], [6, 4], [9, 5], [1, 1]], 2: [[3, 10], [3, 9], [3, 8]], 3: [[3, 15], [2, 15], [1, 15]], 4: [[5, 1], [4, 1], [3, 1]], 5: [[3, 19], [4, 19], [5, 19]], 6: [[6, 14], [7, 14], [8, 14]], 7: [[7, 19], [8, 19], [9, 19]], 'board_width': 20, 'board_height': 10, 'last_direction': None, 'controlled_snake_index': 4}, {1: [[9, 6], [7, 12], [6, 4], [9, 5], [1, 1]], 2: [[3, 10], [3, 9], [3, 8]], 3: [[3, 15], [2, 15], [1, 15]], 4: [[5, 1], [4, 1], [3, 1]], 5: [[3, 19], [4, 19], [5, 19]], 6: [[6, 14], [7, 14], [8, 14]], 7: [[7, 19], [8, 19], [9, 19]], 'board_width': 20, 'board_height': 10, 'last_direction': None, 'controlled_snake_index': 5}, {1: [[9, 6], [7, 12], [6, 4], [9, 5], [1, 1]], 2: [[3, 10], [3, 9], [3, 8]], 3: [[3, 15], [2, 15], [1, 15]], 4: [[5, 1], [4, 1], [3, 1]], 5: [[3, 19], [4, 19], [5, 19]], 6: [[6, 14], [7, 14], [8, 14]], 7: [[7, 19], [8, 19], [9, 19]], 'board_width': 20, 'board_height': 10, 'last_direction': None, 'controlled_snake_index': 6}, 
        # {1: [[9, 6], [7, 12], [6, 4], [9, 5], [1, 1]], 2: [[3, 10], [3, 9], [3, 8]], 3: [[3, 15], [2, 15], [1, 15]], 4: [[5, 1], [4, 1], [3, 1]], 5: [[3, 19], [4, 19], [5, 19]], 6: [[6, 14], [7, 14], [8, 14]], 7: [[7, 19], [8, 19], [9, 19]], 'board_width': 20, 'board_height': 10, 'last_direction': None, 'controlled_snake_index': 7}]

        obs = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:

            # ================================== inference ========================================
            # For each agents i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model.choose_action(obs)

            # ============================== add opponent actions =================================
            # we use rule-based greedy agent here. Or, you can switch to random agent.
            actions = logits_greedy(state_to_training, logits, height, width)
            # actions = logits_random(act_dim, logits)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)

            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            # if done:
            #     if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
            #         step_reward = get_reward(info, ctrl_agent_index, reward, score=1)
            #     elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
            #         step_reward = get_reward(info, ctrl_agent_index, reward, score=2)
            #     else:
            #         step_reward = get_reward(info, ctrl_agent_index, reward, score=0)
            # else:
            #     if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
            #         step_reward = get_reward(info, ctrl_agent_index, reward, score=3)
            #     elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
            #         step_reward = get_reward(info, ctrl_agent_index, reward, score=4)
            #     else:
            #         step_reward = get_reward(info, ctrl_agent_index, reward, score=0)

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, ctrl_agent_index, reward, beans, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, ctrl_agent_index, reward, beans, score=2)
                else:
                    step_reward = get_dense_reward(info, ctrl_agent_index, reward, beans, score=0)
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, ctrl_agent_index, reward, beans, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, ctrl_agent_index, reward, beans, score=4)
                else:
                    step_reward = get_dense_reward(info, ctrl_agent_index, reward, beans, score=0)

            done = np.array([done] * ctrl_agent_num)

            # ================================== collect data ========================================
            # Store transition in R
            # print(obs.shape) (3, 142) -> (num_agents, obs.shape)
            model.replay_buffer.push(obs, logits, step_reward, next_obs, done)

            model.update()

            obs = next_obs
            state_to_training = next_state_to_training
            beans = info['beans_position']
            step += 1

            if args.episode_length <= step or (True in done):

                print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} epsilon: {model.eps:.2f}')
                print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')

                reward_tag = 'reward'
                loss_tag = 'loss'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                    'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})
                if model.c_loss and model.a_loss:
                    writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'actor': model.a_loss, 'critic': model.c_loss})

                if model.c_loss and model.a_loss:
                    print(f'\t\t\t\ta_loss {model.a_loss:.3f} c_loss {model.c_loss:.3f}')

                if episode % args.save_interval == 0:
                    model.save_model(run_dir, episode)

                env.reset()
                beans = env.beans_position.copy()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    main(args)

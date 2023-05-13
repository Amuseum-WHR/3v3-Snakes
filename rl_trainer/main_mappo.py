import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.mappo import MAPPO
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
    obs_dim = 142 # 26
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model = MAPPO(obs_dim, act_dim, ctrl_agent_num, args)

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        
        # step = 0
        # episode_reward = np.zeros(6)

        obs_ep, action_ep, reward_ep, next_obs_ep, done_ep = [], [], [], [], []


        for _ in range(args.batch_size // args.episode_length):

            state = env.reset()
            state_to_training = state[0]
            obs = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)
            beans = env.beans_position.copy()
            episode += 1
            step = 0
            episode_reward = np.zeros(6)

            while step < args.episode_length:
                
                logits = model.choose_action(obs)
                # print(logits.shape)
                actions = logits_greedy(state_to_training, logits, height, width)
                # actions = logits_random(act_dim, logits)

                next_state, reward, done, _, info = env.step(env.encode(actions))
                # print(reward)
                next_state_to_training = next_state[0]
                next_obs = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)

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

                obs_ep.append(obs)
                action_ep.append(logits)
                reward_ep.append(step_reward)
                next_obs_ep.append(next_obs)
                done_ep.append(done)

                obs = next_obs
                state_to_training = next_state_to_training
                beans = info['beans_position'].copy()
                step += 1


            print("eps", episode_reward)

        print(episode)

        actor_loss, critic_loss, mean_reward = model.update((obs_ep, action_ep, reward_ep, next_obs_ep, done_ep))

        writer.add_scalar('actor_loss', actor_loss, episode//6)
        writer.add_scalar('critic_loss', critic_loss, episode//6)
        writer.add_scalar('mean_reward', mean_reward, episode//6)

        if episode % 120 == 0:
            model.save_model(run_dir, episode)

        env.reset()

        

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="mappo", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--lmbda', default=0.95, type=int)
    parser.add_argument('--a_lr', default=1e-5, type=float)
    parser.add_argument('--c_lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2400, type=int)
    parser.add_argument('--epsilon_greedy', default=0.4, type=float)
    parser.add_argument('--epsilon_clip', default=0.2, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    main(args)

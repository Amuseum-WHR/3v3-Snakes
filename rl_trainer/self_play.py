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
from multiprocessing import Pool as ProcessPool


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def rollout(worker):
    return worker.collect_data()



class RolloutWorker:
    def __init__(self, args):
        self.args = args
        self.env = make(args.game_name, conf=None)
        self.num_agents = self.env.n_player
        self.ctrl_agent_index = [0, 1, 2]
        self.opp_agent_index = [3, 4, 5]
        self.ctrl_agent_num = len(self.ctrl_agent_index)

        self.width = self.env.board_width
        self.height = self.env.board_height

        self.act_dim = self.env.get_action_dim()
        self.obs_dim = 122
        self.model = MAPPO(self.obs_dim, self.act_dim, self.ctrl_agent_num, args)
        self.eps = 0

        run_dir, log_dir = make_logpath(self.args.game_name, self.args.algo)

        if args.load_model:
            load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
            self.model.load_model(load_dir, episode=args.load_model_run_episode)

    def collect_data(self):
        # print(self.eps)
        state = self.env.reset()
        state_to_training = state[0]
        obs = get_observations(state_to_training, self.ctrl_agent_index, self.obs_dim, self.height, self.width)
        obs_opp = get_observations(state_to_training, self.opp_agent_index, self.obs_dim, self.height, self.width)
        beans = self.env.beans_position.copy()
        step = 0
        episode_reward = np.zeros(6)

        obs_ep, action_ep, reward_ep, next_obs_ep, done_ep, global_state_ep = [], [], [], [], [], []

        while step < self.args.episode_length:
            global_state_ep = get_global_state(state_to_training)
            # logits = self.model.choose_action(obs)
            logits = self.model.greedy_init(obs, state[:3], self.eps)

            logits_opp = self.model.greedy_init(obs_opp, state[3:], self.eps)

            actions = np.zeros(6)
            actions[:3] = logits
            actions[3:] = logits_opp
            # actions = action_more_greedy(state[3:], logits, self.height, self.width)
            # actions = action_greedy(state_to_training, logits, height, width)

            next_state, reward, done, _, info = self.env.step(self.env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, self.ctrl_agent_index, self.obs_dim, self.height, self.width)
            next_obs_opp = get_observations(next_state_to_training, self.opp_agent_index, self.obs_dim, self.height, self.width)
            reward = np.array(reward)

            episode_reward += reward
            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, self.ctrl_agent_index, reward, beans, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, self.ctrl_agent_index, reward, beans, score=2)
                else:
                    step_reward = get_dense_reward(info, self.ctrl_agent_index, reward, beans, score=0)
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, self.ctrl_agent_index, reward, beans, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_dense_reward(info, self.ctrl_agent_index, reward, beans, score=4)
                else:
                    step_reward = get_dense_reward(info, self.ctrl_agent_index, reward, beans, score=0)

            done = np.array([done] * self.ctrl_agent_num)

            obs_ep.append(obs)
            action_ep.append(logits)
            reward_ep.append(step_reward)
            next_obs_ep.append(next_obs)
            done_ep.append(done)

            obs = next_obs
            obs_opp = next_obs_opp
            state_to_training = next_state_to_training
            state = next_state
            beans = info['beans_position'].copy()
            step += 1


        # print("eps", episode_reward)
        return obs_ep, action_ep, reward_ep, next_obs_ep, done_ep, global_state_ep, episode_reward

    def update_model(self, model):
        self.model.actor.load_state_dict(model.actor.state_dict())
        self.model.critic.load_state_dict(model.critic.state_dict())
        self.eps += 1




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
    obs_dim = 122 # 142 # 26
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    eps = 0

    workers = [RolloutWorker(args) for _ in range(args.num_workers)]

    episode = 0

    pool = ProcessPool(processes=args.num_workers)

    model = MAPPO(obs_dim, act_dim, ctrl_agent_num, args)



    while episode < args.max_episodes:

        res = pool.map(rollout, workers)
        obs_ep, action_ep, reward_ep, next_obs_ep, done_ep, global_state_ep, episode_reward = zip(*res)

        episode += 1

        actor_loss, critic_loss, mean_reward = model.update((obs_ep, action_ep, reward_ep, next_obs_ep, done_ep, global_state_ep))

        # print(2)

        [w.update_model(model) for w in workers]
        episode_reward = np.array(episode_reward)
        my_score, opp_score = np.mean(np.sum(episode_reward[:, :3], axis=1)), np.mean(np.sum(episode_reward[:, 3:], axis=1))
        print(my_score, opp_score)

        # print(3)

        writer.add_scalar('actor_loss', actor_loss, episode)
        writer.add_scalar('critic_loss', critic_loss, episode)
        writer.add_scalar('mean_reward', mean_reward, episode)
        writer.add_scalar('my_score', my_score, episode)
        writer.add_scalar('opp_score', opp_score, episode)

        if episode % 100 == 0:
            model.save_model(run_dir, episode)

        eps += 1

        # env.reset()






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
    parser.add_argument('--a_lr', default=2e-5, type=float)
    parser.add_argument('--c_lr', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=4800, type=int)
    parser.add_argument('--epsilon_greedy', default=0, type=float)
    parser.add_argument('--epsilon_clip', default=0.2, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--num_workers', default=24, type=int)

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')


    main(args)

from trainer import Trainer
import gym
import torch
from model import G2RL
import random
import numpy as np
from torch.optim import RMSprop
from create_env_train import MAPFEnv
import argparse
from config import Config
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append((s0, a, r, s1, done))

    def sample(self, batch_size):

        #s0, agent, reward
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        return (s0), a, r, (s1), done

    def size(self):
        return len(self.buffer)

class G2RLAgent:
    def __init__(self, input_embedding = 256, maxbuff = 10, pre_train = False, use_cuda = False ):
        self.use_cuda = use_cuda
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(capacity=maxbuff)

        self.model = G2RL(input_embedding)
        self.target_model = G2RL(input_embedding)
        if pre_train :
            self.target_model.load_state_dict()
        else:
            self.target_model.init_xavier()
        self.model_optim = None

    def act(self, state, epsilon=None):
        try:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        except:
            print(state)
        if self.config.use_cuda:
            state = state
        q_value = self.model.forward(state)
        action = q_value.max(1)[1].item()


        return action



    def learning(self, fr, lr = 0.0001, update_tar_interval = 1):
        #optim method la RMS prop thep paper
        self.model_optim = RMSprop(self.model.parameters(), lr=lr)
        #s0, agent, reward
        s0, a, r, s1, done = self.buffer.sample(8)


        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

        q_values = self.model(s0)
        next_q_values = self.model(s1)
        next_q_state_values = self.target_model(s1)

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)

        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        self.model_optim.zero_grad()

        # backprop
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        loss.backward()
        self.model_optim.step()

        return loss.item()

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def load_weights(self, model_path):
        if model_path is None: return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='CartPole-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()
    # ddqn.py --train --env CartPole-v0

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.001
    config.eps_decay = 500
    config.frames = 160000
    config.use_cuda = True
    config.learning_rate = 1e-3
    config.max_buff = 1000
    config.update_tar_interval = 100
    config.batch_size = 8
    config.print_interval = 200
    config.log_interval = 200
    config.win_reward = 198
    config.win_break = True

    env = MAPFEnv(PROB=(.3, .4), SIZE=(30, 30), DIAGONAL_MOVEMENT=False, observation_size=16 )

    # config.action_dim = env.action_space.n
    # config.state_dim = env.observation_space.shape[0]
    agent = G2RLAgent()

    trainer = Trainer(agent, env, config)
    trainer.train()

    # elif args.test:
    #     if args.model_path is None:
    #         print('please add the model path:', '--model_path xxxx')
    #         exit(0)
    #     tester = Tester(agent, env, args.model_path)
    #     tester.test()
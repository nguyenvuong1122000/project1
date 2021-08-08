
import gym
import torch
from model import G2RL
import random
import numpy as np
from torch.optim import RMSprop

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done))

    def sample(self, batch_size):
        #s0, agent, reward
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s0), a, r, np.concatenate(s1), done

    def size(self):
        return len(self.buffer)

class G2RLAgent:
    def __init__(self, input_embedding = 256, maxbuff = 10, pre_train = False, use_cuda = False ):
        self.use_cuda = use_cuda
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(capacity=maxbuff)

        self.model = G2RL(input_embedding).cuda()
        self.target_model = G2RL(input_embedding).cuda()
        if pre_train :
            self.target_model.load_state_dict()
        else:
            self.target_model.init_xavier()
        self.model_optim = None

    def learning(self, fr, lr = 0.0001, update_tar_interval = 1):
        #optim method la RMS prop thep paper
        self.model_optim = RMSprop(self.model.parameters(), lr=lr)
        #s0, agent, reward
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)

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

        q_values = self.model(s0).cuda()
        next_q_values = self.model(s1).cuda()
        next_q_state_values = self.target_model(s1).cuda()

        # tính reward đơn giản, chưa theo paper
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)

        # backprop
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def load_weights(self, model_path):
        if model_path is None: return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))


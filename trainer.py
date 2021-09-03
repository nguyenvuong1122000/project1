import math
import random

import numpy as np
from config import Config

class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        # self.outputdir = get_output_folder(self.config.output, self.config.env)
        # self.agent.save_config(self.outputdir)
        # self.board_logger = TensorBoardLogger(self.outputdir)

    def train(self, pre_fr=0):
        for i in range(100):
            losses = []
            all_rewards = []
            episode_reward = 0
            ep_num = 0
            is_win = False
            ep_num = ep_num+1
            self.env._reset(1)
            state = self.env.observe(1)
            for fr in range(50):
                reward_sum = 0
                epsilon = self.epsilon_by_frame(fr)
                action = self.agent.act(state, epsilon)
                next_state, reward, done, _ = self.env._step((1,action))
                self.agent.buffer.add(state, action, reward, next_state, done)

                reward_sum = reward_sum + reward
                state = next_state
                episode_reward += reward

                loss = 0
                if self.agent.buffer.size() > self.config.batch_size:
                    loss = self.agent.learning(fr)
                    losses.append(loss)

                #
                # if fr % self.config.log_interval == 0:
                #     self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])
                #
                # if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                #     self.agent.save_checkpoint(fr, self.outputdir)

                if done:
                    self.env._reset(1)
                    print("done")
                    continue
                #     state = self.env._reset(1)
                #     all_rewards.append(episode_reward)
                #     episode_reward = 0
                #     ep_num += 1
                #     avg_reward = float(np.mean(all_rewards[-100:]))
                #
                #     if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                #         is_win = True
                #         self.agent.save_model(self.outputdir, 'best')
                #         print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                #         if self.config.win_break:
                #             break
            print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (fr, reward_sum/50, loss, ep_num))

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')
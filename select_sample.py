import random
import gym
import numpy as np
import scipy.io as sio
import torch
from scipy import spatial
from torch import nn, optim
from tqdm import tqdm, trange
from sklearn.ensemble import RandomForestClassifier
from transformer_cwgan.utils import *
from transformer_cwgan.transcwgan import *


def split_state(data, samples_per_class):
    n_gen_per_class = args.gen_num_per_class
    data = np.array(data)
    assert len(data[0]) % samples_per_class == 0
    state = []

    for i in range(0, n_gen_per_class, samples_per_class):
        s = data[:, i: i + samples_per_class, :]
        s = s.reshape([-1, data.shape[-1]])
        state.append(s)
    return state


class ENV(gym.Env):
    def __init__(self, state, train_set, validate_set, episode_len=1):
        self.s = state  # 所有的状态
        np.random.shuffle(self.s)
        self.train_set = train_set
        self.validate_set = validate_set
        self.episode_len = episode_len
        self.action_space = gym.spaces.MultiBinary(self.s[0].shape[0])
        # self.observation_space = gym.spaces.Box(low=-10, high=10, shape=[self.s[0].shape[0], 11])
        self.model = RandomForestClassifier()
        self.reset()
        self.index = 0

    def step(self, action):
        a = action[0]
        v = action[1]
        temp_x = []
        temp_y = []
        self.index += 1
        for i, item in enumerate(a):
            if item == 1:
                temp_x.append(self.current[i])
                temp_y.append(self.label[i])
        self.model.fit(np.concatenate([np.array(temp_x).reshape(-1, 11), self.train_set[:, :-1]], axis=0),
                       np.concatenate([np.array(temp_y).reshape(-1, ), self.train_set[:, -1]], axis=0))
        accs = []
        for i in range(5):
            acc = self.model.score(self.validate_set[:, :-1], self.validate_set[:, -1])
            accs.append(acc)
        acc = np.max(accs)
        r = acc - v
        done = self.index >= self.episode_len
        if done:
            self.index = 0
        next_s, l = self.next_state()
        return next_s, r, done, {}

    def reset(self):
        s, _ = self.next_state()
        return s

    def next_state(self):
        temp_s = np.array(random.sample(self.s, 1)).squeeze(0)
        s = temp_s[:, :-1]
        l = temp_s[:, -1]
        self.current = s
        self.label = l
        return s, l

    def render(self):
        pass


class Net(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=8):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_dim * 2, hidden_dim),
                                 nn.LeakyReLU())
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        self.action_layer = nn.Sequential(nn.Linear(hidden_dim, 2),
                                          nn.Softmax(dim=1))
        for m in self.action_layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        self.value_layer = nn.Sequential(nn.Linear(hidden_dim, 1))
        for m in self.value_layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, s):
        out = self.net(s)
        a = self.action_layer(out)
        v = self.value_layer(out)
        return a, torch.mean(v)


def train_drl():
    print('Train DRL...')
    select_num = args.select_num
    gen_data = get_gen_dataset()
    train_set = get_train_data(select_num)
    validate_set, test_set = get_validate_test_data()
    state = split_state(gen_data, args.n_sample_per_state)
    agent = Net().float()
    env = ENV(state, train_set, validate_set, args.max_episode_len)
    clip_param = 0.2 # 0.15
    lr = args.lr_rl
    n_epoch = args.episode_num
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    for epoch in trange(n_epoch):
        state = env.reset()
        rewards = []
        log_probs = []

        for t in range(100):
            # 选择动作
            state_tensor = torch.FloatTensor(state)
            action_probs, value_v = agent(state_tensor)
            action = torch.multinomial(action_probs, 1).squeeze(1)
            log_prob = torch.log(action_probs[action])
            next_state, reward, done, _ = env.step([action.tolist(), value_v])

            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            if done:
                break
        rewards = [sum(rewards[i:]) for i in range(len(rewards))]
        policy_loss = torch.rand(args.n_sample_per_state * args.n_class, 2)
        for t in range(len(rewards)):
            a = rewards[t] - sum(rewards[t + 1:])  # 计算优势
            ratio = torch.exp(log_probs[t] - log_probs[t - 1])
            su = torch.min(ratio * a, torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * a)
            policy_loss -= su
        policy_loss = policy_loss.mean()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
    env.close()


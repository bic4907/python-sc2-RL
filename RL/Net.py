import numpy as np
import torch
import random
import cv2
import torch.nn.functional as F
from torch import nn

MEMORY_SIZE = 100000
GAMMA = 0.99

EPSILON_MIN = 0.001
LEARNING_RATE = 1e-4

class Net(nn.Module):


    def __init__(self, s_dim, a_dim, param_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.param_dim = param_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 16, kernel_size=2, padding=1),
            nn.ReLU(),
        ).cuda()

        self.action_fc = nn.Sequential(
            nn.Linear(576, 1302),
            nn.ReLU(),
            nn.Linear(1302, a_dim),
            nn.ReLU()
        ).cuda()

        self.param_fc = nn.Sequential(
            nn.Linear(576 + a_dim, 1302),
            nn.ReLU(),
            nn.Linear(1302, 2),
            nn.Sigmoid()
        ).cuda()


        self.step = 0

    def cnn_forward(self, s):
        if s.dim() == 3:
            s = s[np.newaxis, :]
        s = torch.Tensor(s).cuda()
        conv_out = self.conv(s)
        flatten = conv_out.view(conv_out.size(0), -1)
        return flatten

    def action_forward(self, cnn_flatten):
        return self.action_fc(cnn_flatten)

    def param_forward(self, cnn_output, action):

        action = torch.LongTensor(action).reshape([-1, 1]).cuda()

        onehot_action = torch.zeros(len(action), self.a_dim).cuda()
        onehot_action.scatter_(1, action, 1.0)

        input = torch.cat((cnn_output, onehot_action), -1)
        return self.param_fc(input)


class DQN():
    def __init__(self, s_dim, a_dim):
        self.replay_buffer = []
        self.global_step = 0
        self.global_episode = 0
        self.epsilon = 1

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.m_net = Net(s_dim, a_dim, 2) # 2 is for Positon(x, y)
        self.t_net = Net(s_dim, a_dim, 2)

        self.require_position = [False, False, False, True, True]

        initialize(self.m_net)
        initialize(self.t_net)

        self.optimizer = torch.optim.Adam(self.m_net.parameters(), lr=LEARNING_RATE)

    def update_target(self):
        self.t_net.load_state_dict(self.m_net.state_dict())

    def push_to_buffer(self, s, a, action_r, param_r, s_):
        s = s.reshape(-1)
        a = np.array([a]).reshape(-1)
        action_r = np.array([action_r]).reshape(-1)
        param_r = np.array([param_r]).reshape(-1)
        s_ = s_.reshape(-1)

        transition = np.hstack((s, a, action_r, param_r, s_))
        if self.is_replay_full():
            self.replay_buffer.pop(0)
        self.replay_buffer.append(transition)

    def clear_buffer(self):
        self.replay_buffer.clear()

    def is_replay_full(self):
        return len(self.replay_buffer) == MEMORY_SIZE

    def train(self):
        sample = random.sample(self.replay_buffer, 35)
        action_loss, param_loss = self.get_loss(sample)

        # Update Paramter Network first
        '''
        self.optimizer.zero_grad()
        param_loss.backward()
        self.optimizer.step()
        '''
        # Update Action Network second
        self.optimizer.zero_grad()
        action_loss.backward()
        self.optimizer.step()


    def get_action(self, obs):
        cnn_feature = self.m_net.cnn_forward(np2torch(obs))

        if random.uniform(0, 1) <= self.epsilon:  # Exploration
            rand_a = np.random.randint(0, self.a_dim)
            action = np.array([[rand_a]])
            param = np.random.rand(2)
        else:
            q_value = self.m_net.action_forward(cnn_feature)
            action = torch.argmax(q_value).cpu().detach().reshape([-1, 1])
            param = np.random.rand(2)

        if (self.epsilon > EPSILON_MIN):
            self.epsilon = self.epsilon * 0.999

        return int(action), param

    def get_param(self, cnn_feature, action):
        return self.get_param(cnn_feature, action)

    def get_probs(self, obs):
        cnn_feature = self.m_net.cnn_forward(np2torch(obs))
        q_value = self.m_net.action_forward(cnn_feature)
        #q_softmax = F.softmax(q_value, dim=-1)
        return q_value.detach().cpu().numpy()[0]

    def get_loss(self, memory):
        s_dim = self.s_dim
        memory = np.vstack(memory)

        # Make Batches
        q_batch_s = torch.Tensor(memory[:, 0:7500]).reshape([-1, 3, 50, 50])
        q_batch_a = torch.LongTensor(memory[:, 7500]).reshape([-1, 1])
        q_batch_r = torch.Tensor(memory[:, 7501]).reshape([-1, 1])
        p_batch_r = torch.Tensor(memory[:, 7502]).reshape([-1, 1])
        q_batch_s_ = torch.Tensor(memory[:, 7503:]).reshape([-1, 3, 50, 50])


        m_cnn_features = self.m_net.cnn_forward(q_batch_s)
        t_cnn_features = self.t_net.cnn_forward(q_batch_s_).detach()

        # Make Action loss
        t_q_next = self.t_net.action_forward(t_cnn_features).cpu().max(dim=-1)[0].reshape([-1, 1]).detach()
        q_target = (q_batch_r + GAMMA * t_q_next)
        q_main = self.m_net.action_forward(m_cnn_features).gather(dim=1, index=q_batch_a.cuda()).cpu()
        action_loss = torch.nn.MSELoss()(q_main, q_target)

        #Make Param loss
        '''
        t_p_next = self.t_net.param_forward(m_cnn_features, q_batch_a).cpu().detach()
        p_target = (p_batch_r + GAMMA * t_p_next)
        m_cnn_features = m_cnn_features.detach() # Do not Update CNN
        p_main = self.m_net.param_forward(m_cnn_features, q_batch_a).cpu()
        param_loss = torch.nn.MSELoss()(p_main, p_target)
        '''
        # Return Action loss, Param loss
        return action_loss, None

    def save(self):
        state = {
            'global_episode': self.global_episode,
            'global_step': self.global_step,
            'main_net': self.m_net.state_dict(),
            'target_net': self.t_net.state_dict(),
            'replay_buffer': self.replay_buffer,
            'epsilon': self.epsilon
        }
        torch.save(state, 'save_model/' + ("%07d" % (self.global_episode)) + '.pt')

    def load(self, path):
        data = torch.load('save_model/' + path)
        self.global_episode = data['global_episode']
        self.global_step = data['global_step']
        self.m_net.load_state_dict(data['main_net'])
        self.t_net.load_state_dict(data['target_net'])
        self.replay_buffer = data['replay_buffer']
        self.epsilon = data['epsilon']
        print('Load model : ' + path)

def np2torch(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.2)
        m.bias.data.fill_(0)
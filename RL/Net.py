import numpy as np
import torch
import random
import cv2
from torch import nn

MAX_EPISODE = 1000000000
MEMORY_SIZE = 200
UPDATE_INTERVAL = 100 # copy interval main to target
GAMMA = 0.9
EPSILON = 0.9
LEARNING_RATE = 0.01




class Net(nn.Module):


    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

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
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, a_dim),
            nn.ReLU()
        )

        self.step = 0

    def forward(self, s):
        if s.dim() == 3:
            s = s[np.newaxis, :]
        # multi-channel to single-channel
        convert_dataset = []
        for data in s:
            b, g, r = cv2.split(data.numpy())
            convert_dataset.append([b, g, r])
        s = torch.Tensor(convert_dataset)

        conv_out = self.conv(s)
        flatten = conv_out.view(conv_out.size(0), -1)
        return self.fc(flatten)

class DQN():
    def __init__(self, s_dim, a_dim):
        self.replay_buffer = []

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.m_net = Net(s_dim, a_dim)
        self.t_net = Net(s_dim, a_dim)

        initialize(self.m_net)
        initialize(self.t_net)

        self.optimizer = torch.optim.Adam(self.m_net.parameters(), lr=LEARNING_RATE)

    def update_target(self):
        self.t_net.load_state_dict(self.m_net.state_dict())

    def push_to_buffer(self, traj):
        #print(s, a, r, s_, done)
        self.replay_buffer.append(traj)
        if len(self.replay_buffer) > MEMORY_SIZE:
            self.replay_buffer.pop(0)
        #print(np.array(self.replay_buffer).shape)

    def clear_buffer(self):
        self.replay_buffer.clear()

    def is_replay_full(self):
        return len(self.replay_buffer) >= MEMORY_SIZE

    def train(self):
        sample = random.sample(self.replay_buffer, 5)
        loss = self.get_loss(sample)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, obs):
        if np.random.uniform() > EPSILON or self.step < MEMORY_SIZE:  # Exploration
            action = np.random.randint(0, self.a_dim)
        else:
            #print('default', np2torch(obs).shape)
            q_value = self.m_net.forward(np2torch(obs))
            #print('NO_OP', q_value[0].data, 'TRAIN_SCV', q_value[1].data,  'TRAIN_MARINE', q_value[2].data,  'BUILD_SUPPLYDEPOT', q_value[3].data,  'BUILD_BARRACK', q_value[4].data)
            action = torch.argmax(q_value).detach().numpy()
        return action


    def get_loss(self, memory):
        s_dim = self.s_dim
        memory = np.vstack(memory)
        batch_s = torch.Tensor(memory[:, 0:3675])
        batch_a = torch.LongTensor(memory[:, 3675]).reshape([-1, 1])
        batch_r = torch.Tensor(memory[:, 3676]).reshape([-1, 1])
        batch_s_ = torch.Tensor(memory[:, 3677:-1])
        batch_done = torch.Tensor(memory[:, -1]).reshape([-1, 1])


        q_next = self.t_net.forward(batch_s_).max(dim=-1)[0].reshape([-1, 1]).detach()
        target = (batch_r + GAMMA * q_next)
        main = self.m_net.forward(batch_s).gather(dim=1, index=batch_a)

        return torch.nn.MSELoss()(main, target)





def np2torch(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
'''
if __name__ == '__main__':
    env = gym.make(ENV_NAME).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    #writer = SummaryWriter()

    main_net = Net(state_dim, action_dim)
    target_net = Net(state_dim, action_dim)
    optimizer = torch.optim.Adam(main_net.parameters(), lr=LEARNING_RATE)

    initialize(main_net)
    initialize(target_net)

    global_step = 0
    global_episode = 0

    rp_memory = []

    while global_step <= MAX_EPISODE:

        s = env.reset()
        epi_reward = 0

        while True:
            a = main_net.get_action(s)
            s_, r, done, _ = env.step(a)
            epi_reward += r

            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            if(global_episode % 10 == 0):
                env.render()


            rp_memory.append(np.hstack((s, a, r, s_, done)))
            if len(rp_memory) > MEMORY_SIZE:
                rp_memory.pop(0)

            if len(rp_memory) >= MEMORY_SIZE:
                sample = random.sample(rp_memory, 36)
                loss = get_loss(sample, main_net, target_net)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % UPDATE_INTERVAL == 0:
                target_net.load_state_dict(main_net.state_dict())

            global_step += 1

            s = s_

            if done:
                break
        global_episode += 1

        #writer.add_scalar('data/reward', epi_reward, global_step)
        if(global_episode % 10 == 0):
            print("%05d\t%07d\t%5.0f" % (global_step, global_episode, epi_reward))
'''
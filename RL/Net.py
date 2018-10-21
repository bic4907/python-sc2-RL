import numpy as np
from torch import nn

MAX_EPISODE = 1000000000
MEMORY_SIZE = 2000
UPDATE_INTERVAL = 100 # copy interval main to target
GAMMA = 0.9
EPSILON = 0.9
LEARNING_RATE = 0.01

class Net(nn.Module):

    global_step = None


    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.net = nn.Sequential(
            nn.Linear(s_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(100, a_dim)
        )

        self.global_step = 0

    def get_action(self, s):

        if np.random.uniform() > EPSILON or self.global_step < MEMORY_SIZE:  # Exploration
            action = np.random.randint(0, self.a_dim)
        else:
            q_value = self.forward(np2torch(s))
            action = torch.argmax(q_value).detach().numpy()
        return action

    def forward(self, s):
        return self.net.forward(s)


def get_loss(memory, main_net, target_net):

    memory = np.vstack(memory)
    batch_s = torch.Tensor(memory[:, :state_dim])
    batch_a = torch.LongTensor(memory[:, state_dim]).reshape(-1, 1)
    batch_r = torch.Tensor(memory[:, state_dim + 1])
    batch_s_ = torch.Tensor(memory[:, state_dim + 2 : -1])


    q_next = target_net.forward(batch_s_).max(dim=-1)[0].detach()

    target = (batch_r + GAMMA * q_next).reshape([-1, 1])

    main = main_net.forward(batch_s).gather(1, batch_a)

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
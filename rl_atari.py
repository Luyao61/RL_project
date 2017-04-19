# coding=utf-8
import gym
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple
import random
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

SCREEN_HEIGHT = 210
SCREEN_WIDTH = 160
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()
NUM_EPISODES = 1000


env = gym.make('Breakout-v0').unwrapped

ACTION_SPACE = env.action_space.n

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Model(nn.Module):
    def forward(self, x):
        output = F.relu(self.pool1(self.bn1(self.conv1(x))))
        output = F.relu(self.pool2(self.bn2(self.conv2(output))))
        output = F.relu(self.pool3(self.bn3(self.conv3(output))))
        return self.classifier(output.view(output.size(0), -1))

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.classifier = nn.Linear(384, ACTION_SPACE)


resize = T.Compose([
                    T.ToPILImage(),
                    T.Scale(160, interpolation=Image.CUBIC),
                    T.ToTensor(),
                    ])


def get_screen():
    # torch rgb order, channel, height, width
    screen = env.render(mode='rgb_array')
    # print(type(screen))
    # print(type(resize(screen)))


    '''
    ToPILImage
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL.Image while preserving
    value range.
    '''

    '''
    Scale
    Rescales the input PIL.Image to the given ‘size’. If ‘size’ is a 2-element tuple or list in the order of
    (width, height), it will be the exactly size to scale. If ‘size’ is a number, it will indicate the size of the
    smaller edge. For example, if height > width, then image will be rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge interpolation: Default: PIL.Image.BILINEAR
    '''

    '''
    ToTensor()
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W)
    in the range [0.0, 1.0].
    '''

    return resize(screen).unsqueeze(0)  # convert to size batch*C*H*W


# env.reset()
# plt.figure()
# plt.imshow(get_screen().squeeze(0).permute(
#     1, 2, 0).numpy(), interpolation='none')
# plt.title("sample image")
# plt.show()


model = Model()
memory = ReplayMemory(10000)
optimizer = torch.optim.RMSprop(model.parameters())

if USE_CUDA:
    model = model.cuda()

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done/EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        state = state.cuda()
        var_z = Variable(state, volatile=True)
        # var_z.data = var_z.data.cuda()
        output_z = model(var_z)
        return_z = output_z.data.max(1)[1]
        return return_z.cpu()
    else:
        return torch.LongTensor([[random.randrange(ACTION_SPACE)]])


def optimize_model():
    global last_sync

    if len(memory) < BATCH_SIZE:
        return
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))

    # find the none final state
    non_final_mask = torch.ByteTensor(
        tuple(map(lambda s: s is not None, batch.next_state))
    )

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    if USE_CUDA:
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        non_final_next_states = non_final_next_states.cuda()
        non_final_mask = non_final_mask.cuda()


    state_action_values = model(state_batch).gather(1, action_batch)

    if USE_CUDA:
        next_state_values = Variable(torch.zeros(BATCH_SIZE).cuda())
    else:
        next_state_values = Variable(torch.zeros(BATCH_SIZE))

    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



episode_durations = []

for i_episode in range(NUM_EPISODES):

    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action[0, 0])
        # env.render()
        reward = torch.Tensor([reward])

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            # output = "episode: {episode:f}; loss: {loss:.4f}; reward: {reward: .2f}; avg_max_q: {q: .4f}".format(episode=t+1, loss=e_loss/(t+1), reward=e_reward, q=e_avgQ/(t+1))
            print("eps: %d; duration: %d" % (i_episode, t+1))
            break;

torch.save(model.state_dict(), 'model1_0418.pth')
env.close()

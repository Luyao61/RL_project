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

SCREEN_HEIGHT = 210
SCREEN_WIDTH = 160
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
USE_CUDA = torch.cuda.is_available()

env = gym.make('Pong-v0').unwrapped

ACTION_SPACE = len(env.action_space)

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
    print(type(screen))
    print(type(resize(screen)))


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

    if USE_CUDA:
        return resize(screen).unsqueeze(0).cuda()
    else:
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
        return model(Variable(state, volatile=True)).data.max(1)[1].cpu()
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
        tuple()
    )






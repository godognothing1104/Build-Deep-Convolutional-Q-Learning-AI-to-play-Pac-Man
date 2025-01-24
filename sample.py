#%% md
# Import Libraries
#%%
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from grpc.framework.interfaces.base.utilities import full_subscription
from torch.utils.data import DataLoader, TensorDataset
#%% md
# Creating the architecture of the Neural NetWork
# 
#%%
class Network(nn.Module):
    def __init__(self,action_size,seed= 42):
        super(Network,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=8,stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(in_features=12800,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=256)
        self.fc3 = nn.Linear(in_features=256,out_features=action_size)
    def forward(self,state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#%% md
# Training the AI
#%%
import gymnasium as gym #Environment initialize
import ale_py
gym.register_envs(ale_py)
env = gym.make('MsPacmanDeterministic-v4',full_action_space = False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape:', state_shape)
print('State size:', state_size)
print('Number of actions:', number_actions)
#%%
learning_rate = 5e-4 #Hyperparameters initializing
minibatch_size = 64
discount = 0.99
#%%
from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocessor = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    return preprocessor(frame).unsqueeze(0)

#%%
class Agent():
    def __init__(self,action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        print(action_size)
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
    def step(self,state,action,reward,next_state,done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state,action,reward,next_state,done))
        if len(self.memory) > minibatch_size:
            experiences = random.sample(self.memory,k = minibatch_size)
            self.learn(experiences,discount)
    def act(self,state,epsilon = 0):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(range(int(self.action_size)))
    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Ensure correct shape for discount multiplication
        q_targets = rewards + (discount_factor * self.target_qnetwork(next_states).max(1)[0].unsqueeze(1) * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


#%%
agent = Agent(number_actions) #Initialize the agent
#%% md
# Training
#%%
number_episodes = 2000
maximum_number_timesteps_per_episode = 10000
epsilon_starting_value  = 1.0
epsilon_ending_value  = 0.01
epsilon_decay_value  = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)

for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 500.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break

print("Hello World!")

#%% md
# Visualize
#%%
import cv2
import glob
import io
import base64
import imageio
import numpy as np
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gymnasium as gym

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        resized_frame = cv2.resize(np.array(frame), (608, 400))  # Resize to fixed size
        frames.append(resized_frame)
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    # Save the frames as GIF instead of MP4
    imageio.mimsave('video.gif', frames, format='GIF', fps=30)
    print("GIF saved successfully.")

def show_video():
    gif_list = glob.glob('*.gif')
    if len(gif_list) > 0:
        gif_file = gif_list[0]
        video = io.open(gif_file, 'rb').read()
        encoded = base64.b64encode(video).decode('ascii')
        display(HTML(f'<img src="data:image/gif;base64,{encoded}" style="height: 400px;" autoplay loop>'))
    else:
        print("Could not find GIF")

# Run the functions
show_video_of_model(agent, 'MsPacmanDeterministic-v4')
show_video()

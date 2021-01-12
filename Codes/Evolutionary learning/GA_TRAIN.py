# -*- coding: utf-8 -*-
"""
This script is used to train a range of policy nets
for genetic algo.

"""
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

### Classes ###
class DQN(nn.Module):
    def __init__(self, img_width, img_height, num_actions):
        super().__init__()
        #self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)
        #self.fc2 = nn.Linear(in_features=24, out_features=32)
        #self.out = nn.Linear(in_features=32, out_features=num_actions)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=448, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=num_actions)
        
    def forward(self, t):
        #t = t.flatten(start_dim=1)
        #t = F.relu(self.fc1(t))
        #t = F.relu(self.fc2(t))
        #t = self.out(t)
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = F.relu(self.conv3(t))
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = self.out(t)
        t = F.softmax(t, dim=1)
        return t
    
class Agent():
    def __init__(self, num_actions, device):
        self.current_step = 0
        self.num_actions = num_actions
        self.device = device
    
    def select_action(self, state, policy_net):
        return policy_net(state).argmax(dim=1).to(device)
            
class PacmanEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make("MsPacman-v0").unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None
         

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)        

    def num_actions_available(self):
        return self.env.action_space.n
    
    def get_actions_meaning(self):
        return self.env.get_action_meanings()
    
    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device= self.device)
    
    def take_action2(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return _, reward, self.done, _ 
    
    def take_sample_action(self):
        _, reward, self.done, _ = self.env.step(self.env.action_space.sample())
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    def get_observation_space(self):
        return self.env.observation_space.shape
    
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen
    
    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        
        resize = T.Compose([
            T.ToPILImage()
            , T.Resize((40, 90))
            , T.ToTensor()
            ])
        
        return resize(screen).unsqueeze(0).to(self.device)

### Functions ###
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

def evaluate(em, net, agent):
    em.reset()
    obs = em.get_state()
    reward = 0.0
    while True:
        action = agent.select_action(obs, net)
        obs, r, done, _ = em.take_action2(action)
        reward += r
        obs = em.get_state()
        if done:
            break
    return reward


def mutate_parent(net):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        p.data += NOISE_STD * noise_t
    return new_net

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)
    
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)

    if is_ipython: display.clear_output(wait=True)
    
def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def resetVariables():
    return 0, 0, False

def evaluatePlots(gen,rewards):
    plt.title("Mean rewards per generation")
    plt.xlabel("Generation")
    plt.ylabel("Rewards")
    plt.plot(range(gen+1), rewards)
    plt.show()
    
### Main ###
if __name__ == "__main__":
    ### Hyper-Parameters ###   
    #GA
    NOISE_STD = 0.9
    POPULATION_SIZE = 100
    PARENTS_COUNT = 15
    
    #Hard Constraints
    gen_max = 500 # Maximum number of generations before stopping
    rewardMeanTarget = 3300 # Target mean reward to achieve
    rewardStdTarget = 200 # Std to achieve

    ### Initialization ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = PacmanEnvManager(device)
    agent = Agent(em.num_actions_available(), device)
    policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.num_actions_available()).to(device)
    gen_idx = 0
    
    ### Training Commences ###
    print("----- Policy Net Architecture -----")
    print(policy_net)
    episode_durations = []
    timestep = 0
    episode = 1
    rewards_list = []
    rewardsScore = 0
    updateTarget = False
    print("Training has begun... Please wait...")
    nets = [
        DQN(em.get_screen_height(), em.get_screen_width(), em.num_actions_available())
        for _ in range(POPULATION_SIZE)
    ]
    population = [
        (net, evaluate(em, net, agent))
        for net in nets
    ]
    model_filepath = "./Trained_DQN_GA_Model.pth"
    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        rewards_list.append(reward_mean)
        print("--- Generation "+str(gen_idx)+" Completed ---")
        print("--- Performance Summary ---")
        print("reward_mean:", reward_mean)
        print("reward_std:", reward_std)
        print("reward_max:", reward_max)
        
        evaluatePlots(gen_idx,rewards_list)
        
        if reward_mean > rewardMeanTarget and reward_std < rewardStdTarget:
            print("Solved in %d generations." % gen_idx)
            fittest = population[0]
            policyNet = fittest[0]
            torch.save(policyNet.state_dict(), model_filepath)
            break
        if gen_idx >= gen_max-1:
            print("Completed max gen criteria: %d" % gen_max)
            fittest = population[0]
            policyNet = fittest[0]
            torch.save(policyNet.state_dict(), model_filepath)
            break
        
        # generate next population
        prev_population = population
        population = [population[0]]
        print("\nGenerating next population and training...\n")
        for _ in range(POPULATION_SIZE-1):
            parent_idx = np.random.randint(0, PARENTS_COUNT)
            parent = prev_population[parent_idx][0]
            net = mutate_parent(parent)
            fitness = evaluate(em, net, agent)
            population.append((net, fitness))
        gen_idx += 1
    em.close()

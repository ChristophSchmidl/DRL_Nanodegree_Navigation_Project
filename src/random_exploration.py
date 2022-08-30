from unityagents import UnityEnvironment
import torch
from strategy import EpsilonGreedyStrategy
from agent import Agent
from setup import fill_memory, train, test
from config import Config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using Pytorch version {torch.__version__} on device {device}")

# please do not modify the line below
env = UnityEnvironment(file_name="src/Banana_Linux_NoVis/Banana.x86_64")
#env = UnityEnvironment(file_name="src/Banana_Linux/Banana.x86_64")

######################################################
#       Get some information about the environment
######################################################

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print(f"Brains available: {env.brain_names}") # ['BananaBrain']
print(f"Brain name: {brain_name}") # BananaBrain
print(f"Selected brain: {brain}")
'''
Selected brain:  Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , ,
'''

print(brain.__dict__)
'''
{'brain_name': 'BananaBrain', 'vector_observation_space_size': 37, 
'num_stacked_vector_observations': 1, 'number_visual_observations': 0, 
'camera_resolutions': [], 'vector_action_space_size': 4, 
'vector_action_descriptions': ['', '', '', ''], 
'vector_action_space_type': 'discrete', 'vector_observation_space_type': 'continuous'}
'''


# reset the environment
banana_env = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(banana_env.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = banana_env.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


######################################################
#       Randomly explore the environment
######################################################

EPISODES = 100
episode_returns = []
log_df = pd.DataFrame(columns=['episode', 'return'])

for episode in range(EPISODES):
    banana_env = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = banana_env.vector_observations[0]            # get the current state

    score = 0                                            # initialize the score
    done = False                                         # check if episode has finished

    while not done:                                      # while episode has not finished
        action = np.random.randint(action_size)          # select an action
        banana_env = env.step(action)[brain_name]        # send the action to the environment
        next_state = banana_env.vector_observations[0]   # get the next state
        reward = banana_env.rewards[0]                   # get the reward
        done = banana_env.local_done[0]                  # see if episode has finished
        score += reward                                  # update the score
        state = next_state                               # roll over the state to next time step

    episode_returns.append(score)
    print(f"Episode {episode} return : {score}")

print("Average return: {}".format(np.mean(episode_returns)))
env.close()

######################################################
#                   Plot the results
######################################################



log_df = log_df.append(pd.DataFrame({'episode': np.arange(len(episode_returns)), 'return': episode_returns}))
sns.lineplot(x = "episode", y = "return", data=log_df)
plt.show()


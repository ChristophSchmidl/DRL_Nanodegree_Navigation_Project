from unityagents import UnityEnvironment
import torch
from strategy import EpsilonGreedyStrategy
from config import Config
from model import DoubleDQNAgent
import numpy as np
from collections import deque


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using Pytorch version {torch.__version__} on device {device}")


######################################################
#               Set up the environment
######################################################


# please do not modify the line below
env = UnityEnvironment(file_name="src/Banana_Linux_NoVis/Banana.x86_64")
#env = UnityEnvironment(file_name="src/Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
banana_env = env.reset(train_mode=True)[brain_name]

action_size = brain.vector_action_space_size
state = banana_env.vector_observations[0]
state_size = len(state)

#####################################
#       Setup experiment
#####################################

agent = DoubleDQNAgent(input_state_size=state_size, output_action_size=action_size, 
                        buffer_size=10000, epsilon=1, epsilon_dec=0.99995, epsilon_min=0.05,
                        gamma=0.95, replace=500, learning_rate=0.0001, batch_size=512, 
                        device=device)

agent.load_models()
n_episodes = 50000
eval_num = 100
samp_rewards = deque(maxlen=eval_num)
rewards = []
average_scores = []
print("Start Training")
best_average = 0
all_scores = []

for i_episode in range(n_episodes):
    banana_env = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = banana_env.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    
    print(f"Episode {i_episode}")
    while True:
        action = agent.select_action(state)            # select an action
        banana_env = env.step(action)[brain_name]        # send the action to the environment
        next_state = banana_env.vector_observations[0]   # get the next state
        reward = banana_env.rewards[0]                   # get the reward
        done = banana_env.local_done[0]                  # see if episode has finished
        agent.store_experience(state, action, reward, next_state, done)
        agent.learn()
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            samp_rewards.append(score)
            break
    if True:
        if float(np.mean(samp_rewards)) > best_average:
            agent.save_models()
            best_average = float(np.mean(samp_rewards))
        print(i_episode, "Average Score: {}".format(np.mean(samp_rewards)), "Maximal", max(samp_rewards),"\tEpsilon", agent.epsilon)
        average_scores.append(np.mean(samp_rewards))
        all_scores.append(score)

print("Average Scores: ", average_scores)
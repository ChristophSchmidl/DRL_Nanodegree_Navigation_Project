# Udacity Nanodegree - Deep Reinforcement Learning

<img src="img/banana.gif" width="650">

## Project 1: Navigation

The first project in this Nanodegree is about the navigation of an agent to collect yellow bananas and avoiding the blue ones.


### Project Details

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward.
- `1` - move backward.
- `2` - turn left.
- `3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of **+13 over 100 consecutive episodes**.



### Getting Started

This repository was implemented with Python version Python 3.9.13. The following steps should enable you to reproduce and test the the implementations.

- Create a python virtual environment: ``python -m venv venv``
- Activate the virtual environment: ``source venv/bin/activate`` (if you are using Linux/Ubuntu)
- Upgrade pip: ``pip install --upgrade pip`` (optional)
- Install dependencies from local folder: ``pip install ./python``

After these instructions, everything should be ready to go. However, if you encounter compatibility issues with your CUDA version and Pytorch, then you could try to solve these problems by installing specific Pytorch versions that fit your CUDA version. In my case, I could resolve it using the following command:

``pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html``


### Instructions

After cloning the repository, you can create a virtual environment with ```python3 -m venv venv```. You can then switch into your virtual environment by executing ```source venv/bin/activate```.
The next step is to install all dependencies by executing ```pip install ./python```. The python folder contains all dependencies that pip needs. 

### Results

#### Deep Q-Learning (DQN)

Training plot of the **DQN algorithm**: 

![DQN-Training](plots/Navigation_project_DQNAgent_train.png)


Evaluation of the trained **DQN agent** over 25 episodes with deterministic behavior:
![DQN-Evaluation](plots/Navigation_project_DQNAgent_eval.png)

#### Double Deep Q-Learning (DDQN)

Training plot of the **DDQN algorithm**: 
![DDQN-Training](plots/Navigation_project_DDQNAgent_train.png)

Evaluation of the trained **DDQN agent** over 25 episodes with deterministic behavior:
![DDQN-Evaluation](plots/Navigation_project_DDQNAgent_eval.png)


#### Dueling Deep Q-Learning (Dueling DQN)

Training plot of the **Dueling DQN algorithm**: 
![Dueling-DQN-Training](plots/Navigation_project_DuelingDQNAgent_train.png)

Evaluation of the trained **Dueling DQN agent** over 25 episodes with deterministic behavior:
![Dueling-DQN-Evaluation](plots/Navigation_project_DuelingDQNAgent_eval.png)


#### Dueling Double Deep Q-Learning (Dueling DDQN)

Training plot of the **Dueling Double DQN algorithm**: 
![Dueling-Double-DQN-Training](plots/Navigation_project_DuelingDDQNAgent_train.png)

Evaluation of the trained **Dueling Double DQN agent** over 25 episodes with deterministic behavior:
![Dueling-Double-DQN-Evaluation](plots/Navigation_project_DuelingDDQNAgent_eval.png)

### Conclusion

Every algorithm was able to solve the navigation problem with an average score of 13+ over 100 consecutive episodes.
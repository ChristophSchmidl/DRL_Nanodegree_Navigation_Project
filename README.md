# Udacity Nanodegree - Deep Reinforcement Learning

<img src="img/banana.gif" width="650">

## Project 1: Navigation

The first project in this Nanodegree is about the navigation of an Agent to collect yellow bananas and avoiding the blue ones.


### Project Details

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward.
- `1` - move backward.
- `2` - turn left.
- `3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.



### Getting Started

- Create a python virtual environment: ``python -m venv venv``
- Activate the virtual environment: ``source venv/bin/activate``
- Upgrade pip: ``pip install --upgrade pip``
- Install dependencies from local folder: ``pip install ./python``

After these instructions, everything should be ready to go. However, if you encounter compatibility issues with your CUDA version and Pytorch, then you could try to solve these problems by installing specific Pytorch versions which fit your CUDA version:

``pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html``


### Instructions

After cloning the repository, you can create a virtual environment with ```python3 -m venv venv```. You can then switch into your virtual environment by executing ```source venv/bin/activate```.
The next step is to install all dependencies by executing ```pip install ./python```. The python folder contains all dependencies that pip needs. 

### Benchmark Implementation

If you're interested in about how long it should take, in the solution code for the project, we were able to solve the project in fewer than 1800 episodes.

<img src="img/benchmark_plot.png" width="500">



### Other Resources

- https://mohitd.github.io/2018/12/23/deep-rl-value-methods.html
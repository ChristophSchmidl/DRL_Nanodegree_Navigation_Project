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

This repository was implemented with Python version 3.9.13. The following steps should enable you to reproduce and test  the implementation.

- Create a python virtual environment: ``python -m venv venv``
- Activate the virtual environment: ``source venv/bin/activate`` (if you are using Linux/Ubuntu)
- Upgrade pip: ``pip install --upgrade pip`` (optional)
- Install dependencies from local folder: ``pip install ./python``

After these instructions, everything should be ready to go. However, if you encounter compatibility issues with your CUDA version and Pytorch, then you could try to solve these problems by installing specific Pytorch versions that fit your CUDA version. In my case, I could resolve it using the following command:

``pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html``


### Instructions

After cloning the repository and installing all necessary dependencies, you can train and evaluate the different agents through the command line. The file ``src/main.py`` is using the ``argparse`` library to parse the command line arguments. You can use the following command to see all available arguments:

``python -m src.main --help``

which outputs the following:

```
Deep Q-Learning - Navigation Project

optional arguments:
  -h, --help            show this help message and exit
  -episodes EPISODES    Number of games/episodes to play. Default is 1000.
  -lr LR                Learning rate for the optimizer. Default is 0.0001.
  -gamma GAMMA          Discount factor for update equation
  -epsilon_start EPSILON_START
                        Starting value for epsilon. Default is 1.0.
  -epsilon_min EPSILON_MIN
                        Minimum value for epsilon in epsilon-greedy action selection. Default is 0.01.
  -epsilon_dec EPSILON_DEC
                        Rate of decay for epsilon. Default is 1e-5.
  -buffer_size BUFFER_SIZE
                        Maximum size of memory/replay buffer. Default is 30000.
  -batch_size BATCH_SIZE
                        Batch size for training. Default is 32.
  -update_target UPDATE_TARGET
                        Interval (of steps) for updating/replacing target network. Default is 1000.
  -gpu GPU              GPU: 0 or 1. Default is 0.
  -load_checkpoint LOAD_CHECKPOINT
                        Load model checkpoint/weights. Default is False.
  -model_path MODEL_PATH
                        Path for model saving/loading. Default is data/
  -plot_path PLOT_PATH  Path for saving plots. Default is plots/
  -save_plot SAVE_PLOT  Save plot of eval or/and training phase. Default is True.
  -algo ALGO            You can use the following algorithms: DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent. Default is DQNAgent.
  -eval EVAL            Evaluate the agent. Deterministic behavior. Default is False.
  -visual_env VISUAL_ENV
                        Using the visual environment. Default is False.
```


#### Training

If you want to start training the DQN agents from scratch with default hyperparamters, you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES>``

#### Evaluation

If you want to evaluate the trained agents in non-visual mode (fast), you can use the following command:

- ``python -m src.main -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES> -use_eval_mode``

The above command simply loads the appropriate model weights, sets the epsilon value to 0.0 to enforce a deterministic behavior (no exploration, pure exploitation) and runs the agent in non-visual mode.

If you want to see the trained agents in action in visual mode (slow), you can use the following command:

- ``python src/main.py -algo <AGENT_NAME> -episodes <NUMBER_OF_EPISODES> -use_eval_mode -use_visual_env``


### Report

If you are interested in the results and a more detailed report, please have a look at [report](REPORT.md).
# Project report:  Navigation

## Project structure

The code is structured in the following way:

- ``main.py``: This file contains the main code to train and test the agent from the command line using Argparse. It is possible to train the agent from scratch or load a pre-trained model. The code is also able to evaluate the agent in the environment in different modes, e.g. visual and non-visual mode. Please keep in mind that the visual mode is slower than the non-visual mode.
- ``src/agents.py:`` Contains the implementation of the different deep q-learning agents as follows:
    - ``Agent``: Base class for all agents
    - ``DQNAgent``
    - ``DoubleDQNAgent``
    - ``DuelingDQNAgent``
    - ``DuelingDoubleDQNAgent`` 

- ``src/replay_buffer.py:`` Contains the implementation of the replay buffer that is used by the agents.

- ``src/networks.py:`` Contains the implementation of two different neural networks that are used by the agents. The first one is a simple fully connected network called ``DeepQNetwork`` with Relu activations, MSELoss and RMSprop optimizer. The other network called ``DuelingDeepQNetwork`` is an extension of the ``DeepQNetwork`` that uses a dueling architecture, i.e., it splits the network into two streams, one for the state value function and one for the advantage function. Both of them are simple fully-connected layers. The final Q-values are computed by combining the state value function and the advantage function.

## Learning algorithm

- Implementation details of: Replay Buffer, inheritance structure of agents, DQN, Double DQN, 
- Give chosen hyperparameters


### Hyperparameters

The following **hyperparamters** have been used for all DQN agents:

| Name   | Value  |
|---|---|
|Learning rate (lr)   | 0.0001  |
| Gamma  | 0.99  |
|Epsilon (start)   | 1.0  |
|Epsilon min (stop)   | 0.01 |
|Epsilon decay  | 1e-5 |
|(Replay) Buffer size  | 30000 |
|Batch size  | 32 |
|Update target (network every n steps)  | 1000 |

Before mentioned values are also the default values for the ``main.py`` script. Therefore, you don't have to specify them explicitly if you want to reproduce the results.

### Model architecture

**DeepQNetwork**

The ``DeepQNetwork`` is a simple three-layer fully connected network with Rectified Linear Unit (ReLU) activations, Mean Squared Error Loss(MSELoss) and Root Mean Squared Propagation (RMSprop) optimizer which is an extension of gradient descent. The input layer has 37 nodes for the 37 features with 128 output nodes. The first layer is followed by a ReLU activation. The second layer has 128 input nodes and 64 output nodes and is also followed by a ReLU activation. The third layer has 64 input nodes and 4 output nodes for every possible action. The output layer is the Q-value for each action. The network is implemented in ``src/networks.py``.

The following figure shows the network architecture:

![DeepQNetwork](/img/DeepQNetwork.onnx.png)

Note: ``Gemm`` stands for General matrix multiplication (dense layer/fully connected layer).

**DuelingDeepQNetwork**

The ``DuelingDeepQNetwork`` is an extension of the ``DeepQNetwork`` that uses a dueling architecture, i.e., it splits the network into two streams, one for the state value function and one for the advantage function (see image below). The value function is responsible for finding the value of a given set of states (single, scalar output in this case). The advantage function is responsible for finding the advantage of each action given a state. **Note**: It seems that I forgot the ReLU activation after the second layer as it was done in the DeepQNetwork architecture. It would be interesting to see if the performance of the DuelingDeepQNetwork would improve if I would add the ReLU activation.

![DuelingDeepQNetwork](/img/DuelingDeepQNetwork.onnx.png)



## Plot of rewards

Every algorithm was able to solve the navigation problem with an average score of 13+ over 100 consecutive episodes. See the following plots for the training and evaluation of the different DQN algorithms.

### Deep Q-Learning (DQN)

Training plot of the **DQN algorithm**: 

![DQN-Training](plots/Navigation_project_DQNAgent_train.png)


Evaluation of the trained **DQN agent** over 25 episodes with deterministic behavior:
![DQN-Evaluation](plots/Navigation_project_DQNAgent_eval.png)

### Double Deep Q-Learning (DDQN)

Training plot of the **DDQN algorithm**: 
![DDQN-Training](plots/Navigation_project_DDQNAgent_train.png)

Evaluation of the trained **DDQN agent** over 25 episodes with deterministic behavior:
![DDQN-Evaluation](plots/Navigation_project_DDQNAgent_eval.png)


### Dueling Deep Q-Learning (Dueling DQN)

Training plot of the **Dueling DQN algorithm**: 
![Dueling-DQN-Training](plots/Navigation_project_DuelingDQNAgent_train.png)

Evaluation of the trained **Dueling DQN agent** over 25 episodes with deterministic behavior:
![Dueling-DQN-Evaluation](plots/Navigation_project_DuelingDQNAgent_eval.png)


### Dueling Double Deep Q-Learning (Dueling DDQN)

Training plot of the **Dueling Double DQN algorithm**: 
![Dueling-Double-DQN-Training](plots/Navigation_project_DuelingDDQNAgent_train.png)

Evaluation of the trained **Dueling Double DQN agent** over 25 episodes with deterministic behavior:
![Dueling-Double-DQN-Evaluation](plots/Navigation_project_DuelingDDQNAgent_eval.png)

## Ideas for future work

- Incorporating Prioritized Experience Replay. See the following paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- Incorporating hyperparameter optimization using random search, grid search and Hyperband. See the following paper: [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560)
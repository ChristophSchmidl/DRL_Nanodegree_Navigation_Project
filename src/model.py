import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replaybuffer import ExperienceReplayBuffer
import os


class DQNNetwork(torch.nn.Module):
    '''
    A simple neural network with linear layers (Applies a linear transformation to the incoming data)
    '''

    def __init__(self, input_state_size, output_action_size, device, learning_rate=1e-4, fc1_size=128, fc2_size=64):
        """
        DQNNetwork constructor

        - input_state_size = size of the states
        - output_action_size = size of the actions
        """
        super().__init__()

        self.fc1 = nn.Linear(input_state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_action_size)

        self.to(device)
        self.device = device

        #self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def to_torch(self, x):
        return torch.tensor(x).float().to(self.device)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = self.to_torch(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

    def save_model(self, file_path):
        """
        Function to save model parameters

        Parameters
        ---
        file_path: str
            Location of the file where the model is to be saved
        Returns
        ---
        none
        """
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        """
        Function to load model parameters

        Parameters
        ---
        file_path: str
            Location of the file from where the model is to be loaded
        device:
            Device in use - CPU or GPU
        Returns
        ---
        none
        """
        # Check if file exists
        if os.path.isfile(file_path):
            print("Loading model from file: ", file_path)
            # map_location is required to ensure that a model that is trained on GPU can be run on CPU
            self.load_state_dict(torch.load(file_path, map_location=self.device))
        else:
            print(f"File not found: {file_path}. Continue training from scratch.")



class DoubleDQNAgent(object):

  def __init__(self, input_state_size, output_action_size, buffer_size, epsilon, epsilon_dec, epsilon_min, gamma, replace, learning_rate, batch_size, device):
    self.input_state_size = input_state_size
    self.output_action_size = output_action_size
    self.epsilon = epsilon
    self.epsilon_dec = epsilon_dec
    self.epsilon_min = epsilon_min
    self.gamma = gamma
    self.replace = replace
    self.learning_rate = learning_rate
    self.buffer_size = buffer_size
    self.batch_size = batch_size
    self.exp_counter = 0
    self.learn_step_counter = 0
    self.device = device

    # Neural Networks
    self.prediction_network = DQNNetwork(self.input_state_size, self.output_action_size, self.device, self.learning_rate )
    self.target_network = DQNNetwork(self.input_state_size, self.output_action_size, self.device, self.learning_rate )

    # ReplayBuffer
    self.replay_buffer = ExperienceReplayBuffer(buffer_size=self.buffer_size, device=self.device)

  def save_models(self):
    self.prediction_network.save_model("data/prediction_network.chkpt")
    self.target_network.save_model("data/target_network.chkpt")

  def load_models(self):
    self.prediction_network.load_model("data/prediction_network.chkpt")
    self.target_network.load_model("data/target_network.chkpt")

  def store_experience(self, state, action, reward, next_state, done):
    self.replay_buffer.add_experience(state, action, reward, next_state, done)

  def select_action(self, state):
    #print(f"State: {state} of type {type(state)}")

    if torch.rand(1).item() < self.epsilon:
        self.epsilon *= self.epsilon_dec
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return int(torch.randint(0,self.output_action_size,(1,)).item())
    else:
        return int(torch.argmax(self.prediction_network.forward(state)).item())

  def replace_target_network(self):
    if self.learn_step_counter % self.replace == 0:
      self.target_network.load_state_dict(self.prediction_network.state_dict())
    self.learn_step_counter += 1

  def learn(self):
    if len(self.replay_buffer) < self.batch_size:
      return
    # Replace target network
    self.replace_target_network()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_experiences(self.batch_size)

    # Q Target Values
    max_actions = self.prediction_network.forward(next_state_batch).detach().argmax(dim=1).long()
    next_q_values = self.target_network.forward(next_state_batch).detach().gather(1, max_actions.unsqueeze(1))
    target_q_values = reward_batch + (self.gamma * next_q_values.squeeze(1) * (1-done_batch.float()))
    target_q_values = target_q_values.unsqueeze(1)

    # Q Predicted
    expected_q_values = self.prediction_network(state_batch).gather(1, action_batch.unsqueeze(1))

    # Compute Loss
    loss = F.mse_loss(expected_q_values, target_q_values)
    # Minimize Loss
    self.prediction_network.optimizer.zero_grad()
    loss.backward()
    self.prediction_network.optimizer.step()
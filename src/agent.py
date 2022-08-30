import random
import torch
import torch.nn.functional as F
from model import DQNNetwork
from replaybuffer import Experience, ExperienceReplayBuffer


class Agent():
    def __init__(self, strategy, state_size, action_size, device, discount=0.99, buffer_size=5000, learning_rate=1e-3,
                 train_mode=True):
        self.current_step = 0
        self.strategy = strategy
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.discount = discount

        self.policy_net = DQNNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            learning_rate=learning_rate,
            fc1_size=128,
            fc2_size=64,
            device=self.device)

        self.target_net = DQNNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            learning_rate=learning_rate,
            fc1_size=128,
            fc2_size=64,
            device=self.device)

        # will notify all your layers that you are in eval mode, that way,
        # batchnorm or dropout layers will work in eval mode instead of training mode.
        self.target_net.eval()
        if not train_mode:
            self.policy_net.eval()

        self.memory = ExperienceReplayBuffer(buffer_size, self.device)

    def get_current_epsilon(self):
        return self.strategy.get_exploration_rate(self.current_step)

    def select_action(self, state):
        exploration_rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if exploration_rate > random.random():
            return random.randrange(self.action_size)  # explore
        else:
            # turn off gradient tracking
            with torch.no_grad():
                # pick the action with maximum Q-value as per the policy Q-network
                action = self.policy_net(state)
                return torch.argmax(action).item()  # exploit

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.get_experience_samples_as_tensors(batch_size)

        # get q values of the actions that were taken, i.e calculate qpred;
        # actions vector has to be explicitly reshaped to nx1-vector
        q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1))

        # calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        q_target = self.target_net.forward(next_states).max(
            dim=1).values  # because max returns data structure with values and indices
        q_target[dones] = 0.0  # setting Q(s',a') to 0 when the current state is a terminal state
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        # calculate the loss as the mean-squared error of yj and qpred
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filename):
        self.policy_net.save_model(filename)

    def load_model(self, filename):
        self.policy_net.load_model(filename=filename, device=self.device)

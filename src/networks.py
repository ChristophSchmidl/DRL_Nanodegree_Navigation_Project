import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

##############################################
#               Deep Q-Network 
##############################################
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        ''' Forward pass through the network, returns the output logits'''
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        actions = self.fc3(state)

        return actions
      
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')

        if os.path.isfile(self.checkpoint_file):
            print("Loading model from file: ", self.checkpoint_file)
            # map_location is required to ensure that a model that is trained on GPU can be run on CPU
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
        else:
            print(f"File not found: {self.checkpoint_file}. Continue training from scratch.")


##############################################
#           Dueling Deep Q-Network 
##############################################
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        # Dueling DQN adjustment
        self.V = nn.Linear(64, 1) # Value function: finding the value of a given set of states (single, scalar output)
        self.A = nn.Linear(64, n_actions) # Advantage function: finding the advantage of each action given a state
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)

        # Dueling DQN adjustment
        V = self.V(state)
        A = self.A(state)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')

        if os.path.isfile(self.checkpoint_file):
            print("Loading model from file: ", self.checkpoint_file)
            # map_location is required to ensure that a model that is trained on GPU can be run on CPU
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
        else:
            print(f"File not found: {self.checkpoint_file}. Continue training from scratch.")

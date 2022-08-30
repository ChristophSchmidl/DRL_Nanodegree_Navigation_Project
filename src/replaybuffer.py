import random
import torch
import numpy as np
import pickle
from collections import deque


class ExperienceReplayBuffer:
    '''
    Rolling history of experiences

    '''

    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def add_experience(self, state, action, reward, next_state, done):
        # append right to the dequeue
        self.buffer.append(self._to_torch_tensor(state, action, reward, next_state, done))

    def _to_torch_tensor(self, state, action, reward, next_state, done):
        '''
        Convert a single experience into a torch tensor
        Returns: Tuple of torch tensors
        '''
        state = torch.from_numpy(np.asarray(state)).float().to(self.device)
        action = torch.from_numpy(np.asarray(action)).to(self.device)
        reward = torch.from_numpy(np.asarray(reward)).float().to(self.device)
        next_state = torch.from_numpy(np.asarray(next_state)).float().to(self.device)
        done = torch.from_numpy(np.asarray(done)).to(self.device)

        return (state, action, reward, next_state, done)

    def sample_experiences(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        experiences = random.sample(self.buffer, k=batch_size)
        #print(f"Experiences: {experiences} of type {type(experiences)}")
        state_batch = torch.stack([experience[0] for experience in experiences])
        action_batch = torch.stack([experience[1] for experience in experiences])
        reward_batch = torch.stack([experience[2] for experience in experiences])
        next_state_batch = torch.stack([experience[3] for experience in experiences])
        done_batch = torch.stack([experience[4] for experience in experiences])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save_buffer(self, filename='buffer_experiences.pickle'):
        # See: https://www.techcoil.com/blog/how-to-save-and-load-objects-to-and-from-file-in-python-via-facilities-from-the-pickle-module/
        with open(filename, 'wb') as buffer:
            pickle.dump(self.buffer, buffer)

    def load_buffer(self, filename='src/buffer_experiences.pickle'):
        with open(filename, 'rb') as buffer:
            self.buffer = pickle.load(buffer)

    def __len__(self):
        return len(self.buffer)

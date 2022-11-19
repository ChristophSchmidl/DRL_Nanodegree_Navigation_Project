import numpy as np


class ReplayBuffer:
    '''
    Rolling history of experiences.

    We could have used a deque from the collections module, but it is not
    as efficient as a numpy array. Moreover, we would have to convert the
    deque to a numpy array before feeding it to the neural network anyway.
    '''

    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0
        
        # float32 has the least compatibility issues with Pytorch
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                            dtype=np.float32)
        # Can be used for masking later on
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        # Give me the position of the first un-occupied memory slot
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        # What is the position of the last stored memory slot?
        max_mem = min(self.mem_counter, self.mem_size)
        # Uniformly sample from the memory
        batch = np.random.choice(max_mem, batch_size, replace=False) # replace=False means -> sample and discard

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

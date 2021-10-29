import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, buffer_name):
        self.mem_size = max_size
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.name = '/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/'
        self.name += buffer_name + '/' if len(buffer_name)!=0 else ''

        self.load()

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save(self):
      np.save(self.name + 'counter.npy', np.array(self.mem_cntr))
      np.save(self.name + 'states.npy', self.state_memory)
      np.save(self.name + 'actions.npy', self.action_memory)
      np.save(self.name + 'rewards.npy', self.reward_memory)
      np.save(self.name + 'new_states.npy', self.new_state_memory)
      np.save(self.name + 'dones.npy', self.terminal_memory)

    def reset(self):
      np.save(self.name + 'counter.npy', np.array(0))
      np.save(self.name + 'states.npy', np.zeros((self.mem_size, self.input_shape)))
      np.save(self.name + 'actions.npy', np.zeros((self.mem_size, self.n_actions)))
      np.save(self.name + 'rewards.npy', np.zeros(self.mem_size))
      np.save(self.name + 'new_states.npy', np.zeros((self.mem_size, self.input_shape)))
      np.save(self.name + 'dones.npy', np.zeros(self.mem_size, dtype=np.bool))

    def load(self):
      self.mem_cntr = np.load(self.name + 'counter.npy')
      self.state_memory = np.load(self.name + 'states.npy')
      self.action_memory = np.load(self.name + 'actions.npy')
      self.reward_memory = np.load(self.name + 'rewards.npy')
      self.new_state_memory = np.load(self.name + 'new_states.npy')
      self.terminal_memory = np.load(self.name + 'dones.npy')
      
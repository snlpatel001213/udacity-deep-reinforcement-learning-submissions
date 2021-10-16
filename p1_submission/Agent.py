from numpy.lib.twodim_base import mask_indices
import torch
from Model import *
import numpy as np
import random
import torch.nn.functional as F
from collections import namedtuple, deque
import torch.optim as optim


class DQNAgent:
    def __init__(self, action_size:int, state_size:int) -> None:
        self.action_size =  action_size
        self.state_size = state_size
        self.batch_size = 64
        self.seed = 7
        self.buffer_size = 100000
        self.learning_rate = 6e-4
        self.tau = 1e-3
        self.device =  torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
        #prep models
        self.local_model  = DQNModel(self.action_size, self.state_size).to(self.device)
        self.target_model  = DQNModel(self.action_size, self.state_size).to(self.device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.learning_rate)
        self.RB =  ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        self.i_step = 0
        self.learn_every = 4
        self.gamma = 0.99 #discount rate 

    def step(self, state, action, reward, next_state, done):
        # save to memory
        self.RB.add(state, action, reward, next_state, done)
        self.i_step = (self.i_step + 1)%self.learn_every
        if(self.i_step == 0 and len(self.RB) > self.batch_size):
            experiences =  self.RB.sample()
            self.learn(experiences, self.gamma)
            

    def learn(self,  experiences, gamma):
        """
        learn from experiences
        Args:
            experiences 
            gamma 
        """
        state, action, reward, next_state, done = experiences
        q_target_next = self.target_model(next_state).detach().max(1)[0].unsqueeze(1)
        # print("q_target_next : ", q_target_next)
        # print("dones : ",done.squeeze(), (1-done))
        # print("reward : ", reward)
        q_target = reward + (gamma*q_target_next*(1-done))
        # print("q_target : ", q_target)
        q_expected = self.local_model(state).gather(1,action)
        # print("q_expected : ", q_expected)
        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # copy weights
        self.soft_update(local_model=self.local_model, target_model=self.target_model, tau=self.tau)

    def act(self,state, epsilon=0.01):
        """[summary]
        
        Example:
        dqna_agent =  DQNAgent(4, 37)
        dummy_state = np.random.random(37)
        action=dqna_agent.act(dummy_state)
        print(action)

        Args:
            state 
            epsilon (float, optional) Defaults to 0.01.

        Returns:
            Action
        """
        state_tt = torch.Tensor(state).reshape(-1,self.state_size).to(self.device)
        # call local model to get the action for the state
        self.local_model.eval()
        with torch.no_grad():
            action_tt = self.local_model(state_tt).detach().cpu().numpy()
        self.local_model.train()

        if np.random.random()>epsilon:
            return np.argmax(action_tt)
        else:
            return np.random.choice(self.action_size)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self):
        # saving model after sucessful
        torch.save(self.local_model.state_dict(), 'checkpoint.pth')
    def load(self):
        # load saved model
        self.local_model.load_state_dict(torch.load('checkpoint.pth'))
        self.local_model.eval()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
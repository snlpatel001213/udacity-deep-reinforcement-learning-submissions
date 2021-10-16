* # Project 1: Navigation *Report*

  This report summarizes the learning algorithm, and model architecture used to
  train a RL agent to navigate (and collect bananas!) in a large, square world.

  **Goal:** The goal of training is to allow the agent to receive an average reward (over 100 episodes) of at least +13.

  ## Learning algorithm

  This project uses [Deep Q Network
  (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) to train the agent.

  > Here we use recent advances in training deep neural networks to develop a novel artificial agent, termed a deep Q-network, that can learn successful policies directly from high-dimensional sensory inputs using end-to-end reinforcement learning.

  ### Model architecture

  The Deep Q Network has the following architecture having three four connected
  layers.

  ```
  class DQNModel(Module):
      def __init__(self, action_size:int, state_size:int,n_layer1=64, n_layer2=64, n_layer3=32):
          """[summary]
  
          Args:
              action_size ([type]): [description]
              state_size ([type]): [description]
              n_layer1 (int, optional): [description]. Defaults to 16.
              n_layer2 (int, optional): [description]. Defaults to 16.
          """
          super(DQNModel,self).__init__()
          self.fc1 = Linear(in_features=state_size, out_features=n_layer1, bias=True)
          self.fc2 = Linear(in_features=n_layer1, out_features=n_layer2, bias=True)
          self.fc3 =  Linear(in_features=n_layer2, out_features=n_layer3, bias=True)
          self.fc4 =  Linear(in_features=n_layer3, out_features=action_size, bias=True)
      def forward(self, state : torch.FloatTensor):
          """[summary]
  
          Args:
              state ([type]): [description]
  
          Returns:
              [type]: [description]
          """
          layer1_out =  F.relu(self.fc1(state))
          layer2_out =  F.relu(self.fc2(layer1_out))
          layer3_out =  F.relu(self.fc3(layer2_out))
          layer4_out =  self.fc4(layer3_out)
          return layer4_out
  ```

  ### Hyperparameters

  The various hyperparameters used are as follows:

  ```
  self.batch_size = 64
  self.seed = 7
  self.buffer_size = 100000
  self.learning_rate = 6e-4
  self.tau = 1e-3 # weight update
  self.learn_every = 4
  self.gamma = 0.99 #discount rate 
  ```

  ## Training

  The training score progress is as shown.

  ```
  Episode 100	Average Score: 1.98
  Episode 200	Average Score: 6.54
  Episode 300	Average Score: 9.37
  Episode 400	Average Score: 11.90
  Episode 436	Average Score: 13.04
  Environment solved in 336 episodes!	Average Score: 13.04
  ```

  ![training_navigation.png](training_navigation.png)

  ## Future work

  * Try different network architecture with varied number of layers, and number of neurons.
  * Try different combinations of hyperparameters.
  * Use regularization such as dropout in the network.
  * Use more advanced Q learning algorithms such as double/dueling DQN.

  

import torch
from torch import nn
from torch.cuda import device
from torch.nn import Linear
from torch.nn import Module
import numpy as np 
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    
    

# if __name__ == '__main__':
#     DQNModelInstance = DQNModel(action_size=4, state_size=37).to(device)
#     dummy_state = np.random.random(37)
#     torch_tensor = torch.Tensor(dummy_state).reshape(-1,37).to(device)
#     print(DQNModelInstance(torch_tensor))

    
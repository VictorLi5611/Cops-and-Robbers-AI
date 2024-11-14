import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

# class DQN(nn.Module):

#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)

#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)

# import torch.nn as nn
# import torch.nn.functional as F

# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()

#         self.layer1 = nn.Linear(n_observations, 128)
#         # self.bn1 = nn.BatchNorm1d(128)
#         self.in1 = nn.InstanceNorm1d(128, affine=True)
#         self.dropout1 = nn.Dropout(0.2)

#         self.layer2 = nn.Linear(128, 64)
#         # self.bn2 = nn.BatchNorm1d(64)
#         self.in2 = nn.InstanceNorm1d(64, affine=True)
#         self.dropout2 = nn.Dropout(0.2)

#         self.layer3 = nn.Linear(64, 32)
#         # self.bn3 = nn.BatchNorm1d(32)
#         self.in3 = nn.InstanceNorm1d(32, affine=True)
#         self.dropout3 = nn.Dropout(0.2)

#         self.layer4 = nn.Linear(32, 16)
#         # self.bn4 = nn.BatchNorm1d(16)
#         self.in4 = nn.InstanceNorm1d(16, affine=True)
#         self.dropout4 = nn.Dropout(0.2)

#         self.layer5 = nn.Linear(16, n_actions)

#     def forward(self, x):
#         x = F.leaky_relu(self.dropout1(self.bn1(x) if self.training else self.in1(x)))
#         x = F.leaky_relu(self.dropout2(self.bn2(x) if self.training else self.in2(x)))
#         x = F.leaky_relu(self.dropout3(self.bn3(x) if self.training else self.in3(x)))
#         x = F.leaky_relu(self.dropout4(self.bn4(x) if self.training else self.in4(x)))
#         return self.layer5(x)






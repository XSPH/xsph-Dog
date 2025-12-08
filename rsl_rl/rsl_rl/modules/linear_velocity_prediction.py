# linear velocity prediction
# @Time: 2025.12.1 
# @Author:Asuka

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class LinearVelocityPrediction(nn.Module):
    def __init__(self, lin_input_dim, 
                       lin_output_dim, 
                       lin_hidden_dim=[64,64,64],
                       activation='elu'):
        
        super(LinearVelocityPrediction, self).__init__()

        activation = get_activation(activation)    # 获取激活函数类型

        self.lin_input_dim = lin_input_dim  # 输入维度
        self.lin_output_dim = lin_output_dim
        self.lin_hidden_dim = lin_hidden_dim

        prediction_layers = []
        prediction_layers.append(nn.Linear(lin_input_dim,lin_hidden_dim[0]))
        prediction_layers.append(activation)
        for l in range(len(lin_hidden_dim)):
            if l == len(lin_hidden_dim) - 1:
                prediction_layers.append(nn.Linear(lin_hidden_dim[l], lin_output_dim))
            else:
                prediction_layers.append(nn.Linear(lin_hidden_dim[l], lin_hidden_dim[l + 1]))
                prediction_layers.append(activation)
        self.prediction = nn.Sequential(*prediction_layers)

        print(f"LinearVelocityPrediction MLP: {self.prediction}")

    def forward_vel_pred(self, observations):
        v_pred = self.prediction(observations)
        return v_pred



def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
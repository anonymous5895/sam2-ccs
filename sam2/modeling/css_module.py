import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy import sqrt


class CCS(nn.Module):
    def __init__(
            self,
            maxiter=10,
            entropy_epsilon=0.5,
            ker_halfsize=2,
            n_fixed_layers=10
    ):
        super(CCS, self).__init__()
        self.maxiter = maxiter
        # Fixed paramaters of Gaussian function
        self.sigma = torch.full((1, 1, 1), 5.0, dtype=torch.float, requires_grad=False)
        self.ker_halfsize = ker_halfsize
        # Fixed paramater
        self.entropy_epsilon = entropy_epsilon
        self.tau = 1 * self.entropy_epsilon
        if self.maxiter < 20:
            # Define the initial convolution weights
            self.conv_weight = torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]],
                                            [[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]],
                                            requires_grad=False)

            # Fixed nabla convolution layers
            self.nabla_fixed_layers = nn.ParameterList([nn.Parameter(self.conv_weight.clone(), requires_grad=False) for _ in range(n_fixed_layers)])

            # Fixed div convolution layers
            self.div_fixed_layers = nn.ParameterList([nn.Parameter(torch.tensor([[[[0., -1., 0.],
                                                                            [0., 1., 0.],
                                                                            [0., 0., 0.]],
                                                                            [[0., 0., 0.],
                                                                            [-1., 1., 0.],
                                                                            [0., 0., 0.]]]],
                                                                        requires_grad=False)) for _ in range(n_fixed_layers)])

            # Learnable nabla convolution layers
            self.nabla_learnable_layers = nn.ParameterList([nn.Parameter(self.conv_weight.clone(), requires_grad=True) for _ in range(maxiter - n_fixed_layers)])

            # Learnable div convolution layers
            self.div_learnable_layers = nn.ParameterList([nn.Parameter(torch.tensor([[[[0., -1., 0.],
                                                                                [0., 1., 0.],
                                                                                [0., 0., 0.]],
                                                                                [[0., 0., 0.],
                                                                                [-1., 1., 0.],
                                                                                [0., 0., 0.]]]],
                                                                            requires_grad=True)) for _ in range(maxiter - n_fixed_layers)])

        else:
            self.conv_weight = nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]],
                                                [[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]], requires_grad=False))

            self.div_weight = nn.Parameter(torch.tensor([[[[0., -1., 0.],
                                                [0., 1., 0.],
                                                [0., 0., 0.]],
                                               [[0., 0., 0.],
                                                [-1., 1., 0.],
                                                [0., 0., 0.]]]], requires_grad=False))

    def forward(self, o, vector_field):
        o = torch.squeeze(o, dim=1)  # (B, H, W)

        u = torch.sigmoid(o / self.entropy_epsilon)

        # main iteration
        q = torch.zeros_like(o, device=o.device)
        if self.maxiter < 20:
            for i in range(self.maxiter):
                # 1.star-shape
                if i < len(self.nabla_fixed_layers):
                    u_nabla = F.conv2d(u.unsqueeze(1), weight=self.nabla_fixed_layers[i], stride=1, padding=1)
                else:
                    u_nabla = F.conv2d(u.unsqueeze(1), weight=self.nabla_learnable_layers[i - len(self.nabla_fixed_layers)], stride=1, padding=1)

                q = q - self.tau * (
                    u_nabla[:, 0, :, :] * vector_field[:, :, 1] + u_nabla[:, 1, :, :] * vector_field[:, :, 0]
                )
                q[q < 0] = 0

                # Use the appropriate div convolution
                if i < len(self.div_fixed_layers):
                    Tq = F.conv2d(torch.stack([vector_field[:, :, 1] * q, vector_field[:, :, 0] * q], dim=1),
                                weight=self.div_fixed_layers[i], padding=1)
                else:
                    Tq = F.conv2d(torch.stack([vector_field[:, :, 1] * q, vector_field[:, :, 0] * q], dim=1),
                                weight=self.div_learnable_layers[i - len(self.div_fixed_layers)], padding=1)

                # 2.sigmoid
                u = torch.sigmoid((o - Tq.squeeze(dim=1)) / self.entropy_epsilon)
        else:
            for i in range(self.maxiter):
                # 1.star-shape
                
                u_nabla = F.conv2d(u.unsqueeze(1), weight=self.conv_weight, stride=1, padding=1)
                
                q = q - self.tau * (
                    u_nabla[:, 0, :, :] * vector_field[:, :, 1] + u_nabla[:, 1, :, :] * vector_field[:, :, 0]
                )
                q[q < 0] = 0

                
                Tq = F.conv2d(torch.stack([vector_field[:, :, 1] * q, vector_field[:, :, 0] * q], dim=1),
                            weight=self.div_weight, padding=1)
        
                # 2.sigmoid
                u = torch.sigmoid((o - Tq.squeeze(dim=1)) / self.entropy_epsilon)
        u1 = (o - Tq.squeeze(dim=1)) / self.entropy_epsilon
        return u1.squeeze(0)
    
    def STD_Kernel(self, sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        ker = torch.exp(-(x.float() ** 2 + y.float() ** 2) / (2.0 * sigma ** 2))
        ker = ker / (ker.sum(-1, keepdim=True).sum(-2, keepdim=True) + 1e-15)
        ker = ker.unsqueeze(1)
        return ker


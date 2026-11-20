###########################################
# This file contains the following:
# 1. Linear Transformer Model
# 2. Function for clipping gradient
# 3. Function for generating random data
#
# The notation for linear attention follows
# the paper at https://arxiv.org/pdf/2306.00297.pdf
###########################################


import torch
from torch import nn
import numpy as np
import math
from dataclasses import dataclass

# Define data mode class to easily handle different configurations
@dataclass(frozen=True)
class DataMode:
    name: str
    gamma_shape: float | None = None
    gamma_scale: float | None = None
    t_degrees_freedom: int | None = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definition of a single linear attention unit for linear-regression data
# P is the value matrix
# Q is the product of key,query matrices
# the dimensions of the input are
# B: batch-size of prompts
# N: context length (excluding query)
# d: covariate dimension
# P,Q are d x d matrices
# Z is a B x (N+1) + (d+1) matrix
# Output is also B x (N+1) + (d+1)

# For linear attention, activation = None
# For standard attention, activation(x) = torch.nn.functional.softmax(x, dim = 2)
# For ReLU attention, activation(x) = torch.nn.relu(x)
def attention(P,Q,Z, activation = None):
    B= Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1

    P_full =  torch.cat([P,torch.zeros(1,d).to(device)],dim=0)
    P_full =  torch.cat([P_full,torch.zeros(d+1,1).to(device)],dim=1)
    P_full[d,d] = 1
    Q_full = torch.cat([Q, torch.zeros(1,d).to(device)],dim=0)
    Q_full = torch.cat([Q_full, torch.zeros(d+1,1).to(device)],dim=1)

    A = torch.eye(N+1).to(device)
    A[N,N] = 0
    Attn = torch.einsum('BNi, ij, BMj -> BNM', (Z,Q_full,Z))

    # Exclude the query token before softmax normalization so its probability
    # mass is redistributed across the context tokens.
    # TODO: Double check that this is correct
    if isinstance(activation, nn.Softmax) and activation.dim == 2:
        Attn[:, :, N] = -torch.inf

    if activation is not None:
        Attn = activation(Attn)

    key = torch.einsum('ij, BNj -> BNi', (P_full,Z))
    Output = torch.einsum('BNM,ML, BLi -> BNi', (Attn,A,key))

    return Output / N


# The Linear Transformer module
# n_layer denotes the number of layers
# n_head denotes the number of heads. In most of our experiments, n_head = 1
# d denotes the dimension of covariates
# var denotes the variance of initialization. It needs to be sufficiently small, but exact value is not important
# allparam: contains all the parameters, has dimension n_layer x n_head x 2 x d x d
# For example
# - P matrix at layer i, head j is allparam[i,j,0,:,:]
# - Q matrix at layer i, head j is allparam[i,j,1,:,:]
class Transformer_F(nn.Module):
    def __init__(self, n_layer, n_head, d, var, activation=None):
        super(Transformer_F, self).__init__()
        self.register_parameter('allparam', torch.nn.Parameter(torch.zeros(n_layer, n_head, 2, d, d)))
        with torch.no_grad():
            self.allparam.normal_(0,var)
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation = activation

    def forward(self, Z):
        for i in range(self.n_layer):
            Zi = Z
            residues = 0
            # the forward map of each layer is given by F(Z) = Z + attention(Z)
            for j in range(self.n_head):
                Pij = self.allparam[i,j,0,:,:]
                Qij = self.allparam[i,j,1,:,:]
                residues = residues + attention(Pij, Qij, Zi, activation=self.activation)
            Z = Zi + residues
        return Z
    
    #enforces top-left-dxd-block sparsity on p
    def zero_p(self):
        for i in range(self.n_layer):
            for j in range(self.n_head):
                with torch.no_grad():
                    self.allparam[i,j,0,:,:].zero_()


# The Looped Linear Transformer module.
# It repeatedly applies the same residual-attention block, sharing parameters
# across all loops. The leading allparam dimension is kept at size 1 so that
# the stored P/Q matrices follow the same layout as Transformer_F.
class Transformer_F_Looped(nn.Module):
    def __init__(self, n_loop, n_head, d, var, activation=None):
        super(Transformer_F_Looped, self).__init__()
        self.register_parameter('allparam', torch.nn.Parameter(torch.zeros(1, n_head, 2, d, d)))
        with torch.no_grad():
            self.allparam.normal_(0, var)
        self.n_loop = n_loop
        self.n_head = n_head
        self.activation = activation

    def forward(self, Z):
        for _ in range(self.n_loop):
            Zi = Z
            residues = 0
            for j in range(self.n_head):
                Pij = self.allparam[0, j, 0, :, :]
                Qij = self.allparam[0, j, 1, :, :]
                residues = residues + attention(Pij, Qij, Zi, activation=self.activation)
            Z = Zi + residues
        return Z

    # Enforces top-left-dxd-block sparsity on P for the shared block.
    def zero_p(self):
        for j in range(self.n_head):
            with torch.no_grad():
                self.allparam[0, j, 0, :, :].zero_()

# evaluate the loss of model, given data (Z,y)
def in_context_loss(model, Z, y):
    N = Z.shape[1] - 1
    d = Z.shape[2] - 1
    output = model(Z)
    diff = output[:,N,d] + y

    # Restriction to ensure that differences don't overflow and cause infinite loss
    # clamped_diff = torch.nan_to_num(torch.clamp(diff, -5, 5), nan=0.0)

    # loss = ((clamped_diff)**2).mean()
    loss = ((diff)**2).mean()

    return loss
        
# generate random data for linear regression
# mode: Supports 'normal', 'gamma', 'sphere', and 't'. Of type "DataMode",
#       with configurable Gamma parameters or Student-t degrees of freedom.
# N: number of context examples
# d: dimension of covariates
def generate_data(mode: DataMode = DataMode("normal"), N=20, d=1, B=1000, U=None, D=None):
    W= torch.FloatTensor(B, d).normal_(0,1).to(device)
    X = torch.FloatTensor(B, N, d).normal_(0, 1).to(device)
    X_test = torch.FloatTensor(B,1,d).normal_(0, 1).to(device)
    
    if U is not None:
        U = U.to(device)
        D = D.to(device)
        W= torch.FloatTensor(B, d).normal_(0,1).to(device)
        W = torch.mm(W,torch.inverse(D))
        W = torch.mm(W,U.t())
    
    if mode.name =='sphere':
        X.div_(X.norm(p=2,dim=2)[:,:,None])
        X_test.div_(X_test.norm(p=2,dim=2)[:,:,None])
    elif mode.name == 'gamma':
        # random gamma scaling for X
        gamma_scales = np.random.gamma(shape=mode.gamma_shape, scale=mode.gamma_scale, size=[B,N])
        gamma_scales = torch.Tensor(gamma_scales).to(device)
        gamma_scales = gamma_scales.sqrt()

        # random gamma scaling for X_test
        gamma_test_scales = np.random.gamma(shape=mode.gamma_shape, scale=mode.gamma_scale, size=[B,1])
        gamma_test_scales = torch.Tensor(gamma_test_scales).to(device)
        gamma_test_scales = gamma_test_scales.sqrt()

        # normalize to unit norm
        X.div_(X.norm(p=2,dim=2)[:,:,None])
        X_test.div_(X_test.norm(p=2,dim=2)[:,:,None])

        # scale by gamma
        X.mul_(gamma_scales[:,:,None])
        X_test.mul_(gamma_test_scales[:,:,None])
    elif mode.name == 't':
        degrees_freedom = mode.t_degrees_freedom
        if degrees_freedom is None or degrees_freedom <= 0:
            raise ValueError("t mode requires positive t_degrees_freedom")

        # A multivariate Student-t sample is normal data scaled by a shared
        # inverse chi-square variable for each example.
        chi_square_scales = torch.as_tensor(
            np.random.chisquare(df=degrees_freedom, size=[B, N]),
            dtype=X.dtype,
            device=device,
        )
        chi_square_test_scales = torch.as_tensor(
            np.random.chisquare(df=degrees_freedom, size=[B, 1]),
            dtype=X_test.dtype,
            device=device,
        )
        X.mul_((degrees_freedom / chi_square_scales).sqrt()[:, :, None])
        X_test.mul_((degrees_freedom / chi_square_test_scales).sqrt()[:, :, None])
    elif mode.name == 'normal':
        assert True
    elif mode.name == 'relu':
        return generate_data_relu(N=N, d=d, B=B, hidden_dim=d)
    elif mode.name == 'mlp':
        return generate_data_mlp(N=N, d=d, B=B, hidden_dim=d)
    else:
        assert False
        
    if U is not None:
        X = torch.einsum('ij, jk, BNk -> BNi', (U,D,X))
        X_test = torch.einsum('ij, jk, BNk -> BNi', (U,D,X_test))
        
    y = torch.einsum('bi,bni->bn', (W, X)).unsqueeze(2)
    y_zero = torch.zeros(B,1,1).to(device)
    y_test = torch.einsum('bi,bni->bn', (W, X_test)).squeeze(1)
    X_comb = torch.cat([X,X_test], dim=1)
    y_comb = torch.cat([y,y_zero], dim=1)
    Z = torch.cat([X_comb,y_comb], dim=2)

    return Z.to(device), y_test.to(device)

def generate_data_inplace(Z, U=None, D=None):
    B = Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    X = Z[:,:,0:-1]
    X.normal_(0, 1).to(device)
    W= torch.FloatTensor(B, d).normal_(0,1).to(device)
    if U is not None:
        U = U.to(device)
        D = D.to(device)
        W = torch.mm(W,torch.inverse(D))
        W = torch.mm(W,U.t())
        Z[:,:,0:-1] = torch.einsum('ij, jk, BNk -> BNi', (U,D,X))
        
    Z[:,:,-1] = torch.einsum('bi,bni->bn', (W, Z[:,:,0:-1])) #y update
    y_test = Z[:,-1,-1].detach().clone()
    Z[:,-1,-1].zero_()
    return Z.to(device),y_test.to(device)

def generate_data_sine(N=10, B=1000):
    # Sample amplitude a and phase p for each task
    a = torch.FloatTensor(B).uniform_(0.1, 5).to(device)
    p = torch.FloatTensor(B).uniform_(0, math.pi).to(device)
 
    X = torch.FloatTensor(B, N).uniform_(-5, 5).to(device)
 
    Y = a.unsqueeze(1) * torch.sin(p.unsqueeze(1) + X)
 
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)

    return X, Y

def generate_data_relu(mode='normal', N=20, d=1, B=1000, shape_k=0.1, U=None, D=None, hidden_dim=100):
    # Generate random input data
    X = torch.FloatTensor(B, N, d).normal_(0, 1).to(device)
    X_test = torch.FloatTensor(B, 1, d).normal_(0, 1).to(device)

    # Define a 1-hidden layer ReLU network
    model = nn.Sequential(
        nn.Linear(d, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    ).to(device)
    model[0].weight.data.normal_(0, 0.1)
    model[2].weight.data.normal_(0, 0.1)

    # Generate y values using the ReLU network
    y = model(X.view(-1, d)).view(B, N, 1)
    y_test = model(X_test.view(-1, d)).view(B, 1).squeeze(1)
 
    y_zero = torch.zeros(B, 1, 1).to(device)
    X_comb = torch.cat([X, X_test], dim=1)
    y_comb = torch.cat([y, y_zero], dim=1)
    Z = torch.cat([X_comb, y_comb], dim=2)

    return Z, y_test

def generate_data_mlp(N=20, d=1, B=1000, hidden_dim=100):
    # Generate random input data
    X = torch.FloatTensor(B, N, d).normal_(0, 1).to(device)
    X_test = torch.FloatTensor(B, 1, d).normal_(0, 1).to(device)

    # Define a 1-hidden layer ReLU network
    model = nn.Sequential(
        nn.Linear(d, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, d)
    ).to(device)
    model[0].weight.data.normal_(0, 1)
    model[2].weight.data.normal_(0, 1)

    X_MLP = model(X.view(-1, d)).view(B, N, d)
    X_test_MLP = model(X_test.view(-1, d)).view(B, 1, d)

    W = torch.FloatTensor(B, d).normal_(0,1).to(device)
    y = torch.einsum('bi,bni->bn', (W, X_MLP)).unsqueeze(2)
    y_zero = torch.zeros(B,1,1).to(device)
    y_test = torch.einsum('bi,bni->bn', (W, X_test_MLP)).squeeze(1)
    X_comb= torch.cat([X_MLP,X_test_MLP],dim=1)
    y_comb= torch.cat([y,y_zero],dim=1)
    Z= torch.cat([X_comb,y_comb],dim=2)

    return Z, y_test

# LinearTransformer
Pytorch code for reproducing experiments for the following papers:

[1] [Transformers learn to implement preconditioned gradient descent for in-context learning](https://arxiv.org/abs/2306.00297).  *Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra*  
[2] [Linear attention is (maybe) all you need (to understand Transformer optimization)](https://arxiv.org/abs/2310.01082).  *Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Suvrit Sra, Ali Jadbabaie*


<h2>Experiments for <a href=https://arxiv.org/abs/2306.00297>Transformers learn to implement preconditioned gradient descent for in-context learning</h2>

**'simple demonstration.ipynb'**:
- Training a 3 layer Linear Transformer with SGD/Adam, **covariates have identity covariance**
- Plotting test loss
- Displaying matrices at end of training + distance to identity (similar to Figure 4 of [1])

**'rotation demonstration-Adam.ipynb'**:
- Training a 3 layer Linear Transformer with Adam, **covariates have non-identity covariance** (Adam requires about 100x more steps to converge compared to the identity covariance case)
- Plotting test loss
- Displaying matrices at end of training + distance to identity (similar to Figure 4 of [1])
'rotation demonstration-Adam-p0.ipynb' is similar to 'rotation demonstration-Adam.ipynb', but enforces that the P matrix has top left block = 0

**'variable_L_exp.ipynb'**:
- Compares n-layer linear Transformer against n-step Gradient Descent/ Preconditioned Gradient Descent, for n = 1,2,3,4, for fixed context length N=20

**'variable_N_exp.ipynb':**
- Compares 3-layer linear Transformer against 3-step Gradient Descent/ Preconditioned Gradient Descent, for context length N={2,4,6...20}


**'linear_transformer.py'** contains definition of the Linear Transformer model, along with some other handy functions.

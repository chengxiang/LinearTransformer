# LinearTransformer
Pytorch code for reproducing experiments for the following papers:

[1] [Transformers learn to implement preconditioned gradient descent for in-context learning](https://arxiv.org/abs/2306.00297).  *Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra*  
[2] [Linear attention is (maybe) all you need (to understand Transformer optimization)](https://arxiv.org/abs/2310.01082).  *Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Suvrit Sra, Ali Jadbabaie*

'simple demonstration.ipynb' contains code for
- Training a 3 layer Linear Transformer with SGD/Adam, **covariates have identity covariance**
- Plotting test loss
- Displaying matrices at end of training + distance to identity (similar to Figure 4 of [1])

'rotation demonstration-Adam.ipynb' contains code for
- Training a 3 layer Linear Transformer with Adam, **covariates have non-identity covariance** (Adam requires about 100x more steps to converge compared to the identity covariance case)
- Plotting test loss
- Displaying matrices at end of training + distance to identity (similar to Figure 4 of [1])

'linear_transformer.py' contains definition of the Linear Transformer model, along with some other handy functions.

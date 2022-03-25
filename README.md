# An attempt at Implementing Baydin, Pearlmutter et al (2022)

### Problem Statement: 

Optimization/Training a NN to classify a dataset is typically done quickly with backprop and grad descent. However calculating the gradient using backprop is considered biologically implausable. 

The use of directional derivatives can be used to estimate the gradient of a network for a given sample, using 'n' number of forward passes while using 'n' number of one-hot tangent vectors to probe the output of the network w.r.t to all possible weights. For obvious reasons this is much slower than backprop. 

Baydin, Pearlmutter et al (2022) claims that you can get an estimate of a gradient using directional derivatives without running the network 'n' times, but rather using a single optimization step (ie: a batch).

This is done by using a tangent vector with values sampled from a normal distribution, rather than a one-hot vector to calculate the directional derivative. The directional derivative is then multiplied by the same tangent vector used to generate that value and the product is taken as the gradient. 


### Personal Notes:
- Unsure how this works. Works for MNIST but training is VERY slow.
- Factors that improve training speed: 
  -  Batch size : Larger seems better
  -  As training progresses lower LRs improve stability of training.
  -  

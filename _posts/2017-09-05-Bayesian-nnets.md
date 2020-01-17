---
layout: post
title: What is a Bayesian Neural Network? 
author: Jonathan Gordon
comments: true
---

A Bayesian neural network (BNN) refers to extending standard networks with posterior inference. They arise naturally when priors are placed on the weights of a network. They are relatively simple beasts, and (in my opinion) far more interesting than the model itself is the approximate posterior inference associated with them. 

----

Standard NN training via optimization is (from a probabilistic perspective) equivalent to maximum likelihood estimation (MLE) for the weights. For many reasons this is unsatisfactory. One reason is that it lacks proper theoretical justification from a probabilistic perspective: why maximum likelihood? Why just point estimates? Using MLE ignores any uncertainty that we may have in the proper weight values. From a practical standpoint, this type of training is often susceptible to overfitting, as NNs often do.

One partial fix for this is to introduce regularization. From a Bayesian perspective, this is equivalent to inducing priors on the weights (say Gaussian distributions if we are using L2 regularization). Optimization in this case is akin to searching for MAP estimators rather than MLE. Again from a probabilistic perspective, this is not the right thing to do, though it certainly works well in practice.

The correct (i.e., theoretically justifiable) thing to do is posterior inference [[^1], [^2]], though this is very challenging both from a modelling and computational point of view. BNNs are neural networks that take this approach. In the past this was all but impossible, and we had to resort to poor approximations such as Laplace’s method (low complexity) or MCMC (long convergence, difficult to diagnose). However, lately there have been some super-interesting results on using variational inference to do this [[^3]], and this has sparked a great deal of interest in the area.

BNNs are important in specific settings, especially when we care about uncertainty very much. Some examples of these cases are decision making systems, (relatively) smaller data settings, Bayesian Optimization, model-based reinforcement learning and others.
 

_Originally a [Quora answer](https://www.quora.com/What-is-a-Bayesian-Neural-Network), later a [KDnuggets post](https://www.kdnuggets.com/2017/12/what-bayesian-neural-network.html)_

## References
-----
[^1]: Radford, Neal. Bayesian Learning for Neural Networks. 2012
[^2]: Mackay, David. Bayesian Neural Networks and Density Networks. 1995
[^3]: Blundell, Charles, et al. Weight Uncertainty in Neural Networks. 2015

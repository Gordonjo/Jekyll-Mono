---
layout: post
title: Training Deep Models with Stochastic Backpropagation (Part 1 - Background)
author: Jonathan Gordon
---

Recently I've had to train a few deep generative models with stochastic backpropagation (SBP). Specifically, I've been working with variational autoencoders (VAEs) and Bayesian neural networks (BNNs). Anyone who has read the literature on these training procedures and models knows that the theory seems quite complete. However, in practice I have found that quite a lot of elbow grease and magic is required to get these to work well. I'll share some of my experiences here. Since this is a hefty subject, I'll break this into two segments. In this part, I'll provide some background on the subject. In the next part, I'll walk through an example, and highlight some tips and tricks that really help.

## Deep Generative Models
-----

We are considering the case where we have some complex latent variable model (later we will see how we can consider the weights as our latent variables and extend this to BNN) as shown below.

<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/vae.png" width="15%" height="15%">


Here, \\(x\\) are our inputs, \\(z\\) are the latent variables, and \\(\theta\\) parameterizes the conditional distribution. Usually, we use deep neural networks to map from \\(z\\) to \\(x\\), so \\(\theta\\) will be the parameters of the network. This is a very powerful family of models known as deep generative models (DGMs), even for very simple distributions of \\(z\\). What we would like to be able to do is perform learning (i.e., maximum likelihood or a-posteriori estimation) for \\(\theta\\), and inference for \\(z\\). Unfortunately, the posterior distribution for \\(z\\) is intractable, so doing this is not straightforward.


## Variational Inference for DGMs
-----

The best approach seems to be variational inference (VI). The basic idea with VI is to introduce an approximation to the true posterior, which we will call \\(q\\), and parameterize it with the *variational parameters* - \\(\phi\\). Note that to do this, we must choose some parameteric family for \\(q\\), such as a Gaussian. Having chosen \\(q\\), the idea is to minimize the distance between the true posterior and our approximation. The minimization is over \\(\phi\\) and with respect to some divergence between distributions, such as the KL-divergence. 

What's fantastic about VI is that it allows us to convert infernce from an integration problem (which we pretty much suck at) to an optimization one (which we are awesome at). On the flip-side, there are a few drawbacks: one problem is that we are usually limited to very simple posterior approximations, and the quality of our trained model is directly related to the quality of \\(q\\). Another problem is that posterior inference includes a separate set of variational parameters \\(\phi\\) for every data-point, and therefore needs to be recomputed for every new example we receive.   

Luckily, some papers from a few years ago ([^1], [^2]) fleshed out how we could do this! The main idea is to introduce an inference network, that has the job of approximating the posterior distribution of \\(z\\), which is causing all the trouble. The main advantage of this is that it *ammortizes* posterior inference, so that while the number of latent variables grows linearly with the number of data points, posterior inference now has a fixed computational complexity. If we 




[^1]: Kingma, Diederik P and Welling, Max. Auto-encoding variational Bayes. 2013
[^2]: Rezende, Danilo Jimenez, Mohamed, Shakir, and Wierstra, Daan. Stochastic backpropagation and approximate inference in deep generative models. 2014
---
layout: post
title: Training Deep Models with Stochastic Backpropagation
author: Jonathan Gordon
---

Recently I've had to train a few deep generative models with stochastic backpropagation (SBP). Specifically, I've been working with variational autoencoders (VAEs) and Bayesian neural networks (BNNs). Anyone who has read the literature on these training procedures and models knows that the theory seems quite complete. However, in practice I have found that quite a lot of elbow grease and magic is required to get these to work well. I'll share some of my experiences here.

## Stochastic Backpropagation 
-----

For completeness (and to introduce notation), I'll quickly review what it means to train a deep model with SBP. We are considering the case where we have some complex latent variable model (later we will see how we can consider the weights as our latent variables and extend this to BNN) as shown below.

<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/vae.png" width="15%" height="15%">


Here, \\(x\\) are our inputs, \\(z\\) are the latent variables, and \\(\theta\\) parameterizes the conditional distribution. Usually, we use deep neural networks to map from \\(z\\) to \\(x\\), so \\(\theta\\) will be the parameters of the network. This is a very powerful model, even for very simple distributions of \\(z\\). What we would like to be able to do is perform learning (i.e., maximum likelihood or a-posteriori estimation) for \\(\theta\\), and inference for \\(z\\). Unfortunately, the posterior distribution for \\(z\\) is intractable, so doing this is not straightforward.

Luckily, some papers from a few years ago [^1], [^2] fleshed out how we could do this!




[^1]: Kingma, Diederik P and Welling, Max. Auto-encoding variational Bayes. 2013
[^2]: Rezende, Danilo Jimenez, Mohamed, Shakir, and Wierstra, Daan. Stochastic backpropagation and approximate inference in deep generative models. 2014
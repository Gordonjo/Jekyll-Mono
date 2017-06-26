---
layout: post
title: Training Deep Models with Stochastic Backpropagation (Part 1 - Background)
author: Jonathan Gordon
---

Recently I've had to train a few deep generative models with stochastic backpropagation (SBP). Specifically, I've been working with variational autoencoders (VAEs) and Bayesian neural networks (BNNs). Anyone who has read the literature on these training procedures and models knows that the theory seems quite complete. However, in practice I have found that quite a lot of elbow grease and magic is required to get these to work well. I'll share some of my experiences here. Since this is a hefty subject, I'll break this into two segments. In this part, I'll provide some background on the subject. In the next part, I'll walk through an example, and highlight some tips and tricks that really help.

## Deep Generative Models
-----

We are considering the case where we have some complex latent variable model (later we will see how we can consider the weights as our latent variables and extend this to BNN) as shown below.

<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/vae.png" width="20%" height="20%">


Here, \\(x\\) are our inputs, \\(z\\) are the latent variables, and \\(\theta\\) parameterizes the conditional distribution. Usually, we use deep neural networks to map from \\(z\\) to \\(x\\), so \\(\theta\\) will be the parameters of the network. This is a very powerful family of models known as deep generative models (DGMs), even for very simple distributions of \\(z\\). What we would like to be able to do is perform learning (i.e., maximum likelihood or a-posteriori estimation) for \\(\theta\\), and inference for \\(z\\). Unfortunately, the posterior distribution for \\(z\\) is intractable, so doing this is not straightforward.


## Variational Inference for DGMs
-----

The best approach seems to be variational inference (VI). The basic idea with VI is to introduce an approximation to the true posterior, which we will call \\(q\\), and parameterize it with the *variational parameters* - \\(\phi\\). Note that to do this, we must choose some parameteric family for \\(q\\), such as a Gaussian. Having chosen \\(q\\), the idea is to minimize the distance between the true posterior and our approximation. The minimization is over \\(\phi\\) and with respect to some divergence between distributions, such as the KL-divergence. The variational objective can be expressed as:

\begin{equation}
\mathcal{L}(\theta, \phi; x) = \mathbb{E}[\log p(x|z)] - KL(q(z|x)||p(z))
\end{equation}

which has a nice interpretation: the first term can be seen as reconstruction error - we are trying to maximize the likelihood of the data under the latent variable. The second term can be interpreted as a regularizer - it encourages the approximation to be as simple as possible by penalizing it for being different from the prior on \\(z\\), which is typically chosen to be something simple like a standard normal distribution.

What's fantastic about VI is that it allows us to convert infernce from an integration problem (which we pretty much suck at) to an optimization one (which we are awesome at). On the flip-side, there are a few drawbacks: one problem is that we are usually limited to very simple posterior approximations, and the quality of our trained model is directly related to the quality of \\(q\\). Another problem is that posterior inference includes a separate set of variational parameters \\(\phi\\) for every data-point, and therefore needs to be recomputed for every new example we receive.   


## Inference Networks and Reparameterization
-----

To get around these problems, some guys had the brilliant idea of introducing an *inference network*. The idea is to use a neural network to learn \\(q\\). Ideally, we would then be able to train both networks \\(\theta, \phi\\) jointly. The main advantage of this is that it *ammortizes* posterior inference, so that while the number of latent variables grows linearly with the number of data points, posterior inference now has fixed computational and statistical complexity. This seems great, but of course, there are still some problems.

To train the things, we would like to use our regular stochastic gradient optimizers. The KL term in Eq. (1) can often be evaluated analytically, especially for the standard choices of distributions. The difficult term is the reconstruction error, which is pesky because it is under expectation of the intractable posterior. However, we can use Monte-Carlo to approximate it:

\begin{equation}
\mathbb{E}[\log p(x|z)] = \int q(z|x) \log p(x|z)dz \approx \frac{1}{L}\sum\limits_{l=1}^L \log p(x|z^l)
\end{equation} 

where we are sampling \\(z^l \sim q\\). Alas, the problem is that sampling for \\(z\\) means that \\(\hat{\mathcal{L}}\\) is not differentiable w.r.t. \\(\phi\\), so we cannot train the inference network. Another idea might be to directly sample for the gradient \\(\nabla_{\theta, \phi}\mathcal{L}\\). Unfortunately, in [^1] it is shown that \\(\hat{\nabla}_{\phi}\mathcal{L}, while unbiased, has very high variance, and training as such tends to diverge more often than not. 

Luckily, some papers from a few years ago ([^2], [^3]) fleshed out how we could get around this using an almost embarrasingly simple trick - *reparameterization*. The idea here is to introduce a random variable \\(\epsilon\\) who will contain all of the randomness for us. We now set up a new variable \\(\tilde{z}\\) such that:

\begin{equation}
\tilde{z} = g_{\phi}(\epsilon, x);\ \ \tilde{z} \sim q(z|x);\ \ \epsilon \sim p(\epsilon)
\end{equation}

so all we need is that \\(g_{\phi}\\) be differentiable and be able to sample from \\(p_{\epsilon}\\). This is almost always possible, and in many cases it is even super-easy. For example, if:

\begin{equation}
q(z|x) = \mathcal{N}(z; \mu_z, \sigma^2_z)
\end{equation}

Reparameterization, simple though it may appear, does two amazing things. (1) It 

 \rightarrow \epsilon \sim \mathcal{N}(0,1) \text{ and } g_{\phi} = \mu_z + \epsilon \times \sigma_z

## References
-----

[^1]: Paisley, John, Blei, David M, and Jordan, Michael I. Variational Bayesian Inference with Stochastic Search. 2012
[^2]: Kingma, Diederik P and Welling, Max. Auto-encoding variational Bayes. 2013
[^3]: Rezende, Danilo Jimenez, Mohamed, Shakir, and Wierstra, Daan. Stochastic backpropagation and approximate inference in deep generative models. 2014
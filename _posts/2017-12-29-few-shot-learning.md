---
layout: post
title: Recent Advances in Few-Shot Learning
author: Jonathan Gordon
comments: true
---

Few-shot learning is (in my opinion) one of the most interesting and important research areas in ML. It touches at the very core of what ML/DL can’t do today, and is one of the clear indicators of how much work is left. All the approaches to few-shot learning I am aware of use probabilistic modelling. In fact, one of the reasons I’m so sold on Bayesian learning is that it offers avenues to pursue these problems - its not clear to me how one can even approach this not from a probabilistic perspective.

## A Discriminative Approach
----

An interesting paper by Rich Turner's group (in collaboration with Bernard Scholkopf's group) [[^1]] proposed using the representations found in the final hidden layer of a deepNet to generalize to unseen classes for classification.

{:refdef: style="text-align: center;"}
<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/kshot_discriminative.png" width="100%" height="100%">
{:refdef}

By placing a prior on the weights they can perform posterior inference on the values for a new class, and show impressive results using very simple priors and likelihood functions. This is the only discriminative approach I am familiar with, and I think it shows a lot of promise for future work.

## Generative Models for Few-Shot Learning
----

More common are the use of generative models. These can largely be placed on a spectrum of how _structured_ the models are. 

### Unstructured Models

On one end of the generative spectrum are completely unstructured models [[^2], [^3]]. Here, researchers attempt to learn meaningful representations in latent variables, and then make use of that ‘information’ for new tasks. This has shown good results, even though optimal mechanisms for information sharing are not yet clear. A lot of interesting work is being done here.

### Heavily Structured Models

On the other hand are heavily structured models [[^4], [^5], [^6]]. These works come from a faction of ML researchers who believe we should encode as much knowledge as possible into our models [[^7]]. For example, in [[^5]] the authors show how a heavily engineered model can transfer its capabilities to unseen scenarios in game playing, something that state of the art deep RL agents are catastrophically bad at.

{:refdef: style="text-align: center;"}
<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/breakout.png" width="100%" height="100%">
{:refdef}

There is a tradeoff here: structured models achieve very impressive generalization, sample efficiency, and few-shot capabilities. It is clear that by encoding knowledge we can achieve significant gains in these aspects. On the other hand, this entails expert knowledge and the ability to encode it in a computationally tractable manner. This is obviously not possible for all domains, and it may not be a scalable approach to improving AI [[^8]]. Conversely, unstructured generative modelling relieves the burden of human engineering, which is a clear goal of ML. However, it is not clear how to guide the latent variables towards useful representations, nor is it clear what is the ‘right’ way to then transfer this knowledge. 

One interesting recent paper [[^9]] proposes a method for encoding known axes of variation in partially observed variables while allowing latent variables to capture unknown axes of variation. The paper proposes a general approach for variational inference in these models. I believe this approach of finding the middle ground of the spectrum will be very influential in the near future.

_Originally a [Quora answer](https://www.quora.com/What-are-the-recent-developments-in-one-shot-learning/answer/Jonathan-Gordon-23)_

## References
-----

[^1]: Bauer, Matthias, et al. Discriminative K-Shot Learning Using Probabilistic Models. 2017
[^2]: Rezende J., Danilo, et al. One-Shot Generalization in Deep Generative Models. 2016
[^3]: Edwards, Harrison, and Storkey, Amos. Towards a Neural Statistician. 2017
[^4]: Salakhutdinov, Ruslan, Tenenbaum, Joshua, Torralba, Antonio. One-Shot Learning with a Hierarchical Nonparametric Bayesian Model. 2012
[^5]: Kansky, Ken, et al. Schema Networks: Zero-Shot Transfer with a Generative Causal Model of Intuitive Physics. 2017
[^6]: George, Dileep, et al. A Generative Vision Model that Trains with High Data-Efficiency and Breaks Text-Based CAPTCHAs. 2017
[^7]: Lake M., Brenden, et al. Building Machines that Learn and Think Like People. 2017
[^8]: Botvinick, Matthew, et al. Building Machines that Learn and Think for Themselves: Commentary on Lake, Ullman, Tenebaum, and Gershamn. 2017 
[^9]: Siddarth, N., et al. Learning Disentangled Representations with Semi-Supervised Deep Generative Models. 2017

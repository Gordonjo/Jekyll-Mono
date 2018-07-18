---
layout: post
title: Decision-Theoretic Meta-Learning
author: Jonathan Gordon
comments: true
---

In this post I discuss a novel, probabilistic view of meta-learning, and _**Versa**_, a system developed in this framework for 
versatile and efficient few-shot learning. This is joint work with my collaborators John Bronskill, Matthias Bauer, Sebastian
Nowozin, and Richard Turner.

## Meta-Learning

Meta-learning ([^1]) is an old concept in the machine learning research community that has recently been receiving renewed
attention (like most things, Schmidhuber had already thought about it in the 80's ([^2])). The idea is that we can train 
models to _learn how to learn_. The main goal here is to enable models to leverage their learning experiences on previous 
datasets/tasks to quickly learn in the face of new datasets. "Quickly" can mean either computational convergence, or given 
less training data. The idea of meta-learning is very appealing given the massive datasets and compute power generally 
required to train deep learning models.

A straightforward example of such a system was proposed by Ravi and Larochelle ([^3]): they introduced a recurrent neural 
network that learned to output the gradient steps required to train a local model for related but independent datasets. 
Here the _meta-learner_ is the LSTM that learns how to train local learners (say, CNNs) for related image datasets. The 
idea is pretty simple, but works quite nicely.



dfsadas




## References
[^1]: S. Thrun and L. Pratt. Learning to learn. Springer Science & Business Media, 2012.
[^2]: J. Schmidhuber. Evolutionary principles in self-referential learning. PhD thesis, Technische Universität München, 1987.
[^3]: S. Ravi and H. Larochelle. Optimization as a model for few-shot learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.
   


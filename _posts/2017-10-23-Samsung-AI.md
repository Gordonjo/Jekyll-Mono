---
layout: post
title: Samsung Global AI Forum
author: Jonathan Gordon
comments: true
---

Recently I was fortunate enough to be invited to the first annual Samsung Global AI Forum. Samsung recently initiaited a wide range of research collaborations with leading universities around the world as part of their goal of becoming a leader in AI technologies. The forum was the kickoff event to these collaborations. 

The event was great - Samsung hosted us very generously in NYC, and the forum was held at [Samsung 837](https://www.samsung.com/us/837/) which is a very cool venue. There were some great speakers from the participating universities who gave some very interesting talks and panel discussions. Finally, it was a good opportunity for me to meet and interact with other young grad students and hear about the work they're doing and what interests them. Below I'll briefly talk about some of the talks I found especially interesting.

## [Kyunghyun Cho (NYU)](http://www.kyunghyuncho.me/)

Dr Cho gave a very interesting talk at the NLP/U workshop on multi-lingual neural machine translation (NMT) based on [this paper](http://www.sciencedirect.com/science/article/pii/S0885230816301097). In general, Kyunghyun has been a major contributor to the field of NMT, authoring a number of [papers](https://arxiv.org/pdf/1409.0473.pdf) that instigated and shaped the field. In his talk he outlined how one can approach multi-lingual translation by endowing each target/source language with its own encoder/decoder, but using a shared attention mechanism across all languages. This idea is very broad and general, potentially with applications to transfer and multi-modal learning as well. The idea of using a shared attention mechanism is an elegant way of allowing a single network to perform multiple tasks (in this case, switching between multiple languages for translation).

## [Florian Metze (CMU)](http://www.cs.cmu.edu/~fmetze/interACT/Home.html)

Dr Metze presented some recent work of his on improving speech recognition by introducing other modalities as inputs (more details [here](http://ieeexplore.ieee.org/abstract/document/7953112/)). I like this line of research a lot because it seems like a very natural idea (clearly there is much information about what a person is saying in the context of his surroundings, body language, etc.), but in practice there are many difficulties associated with these approaches. The project Florian described highlights how one can use CTC networks to train a network with convolutional features end-to-end and improve overall speech recognition performance.

## [Rob Fergus (NYU)](http://cs.nyu.edu/~fergus/pmwiki/pmwiki.php)

Dr Fergus introduced a super-excting idea about pre-training RL agents in a self-supervised manner ([paper](https://arxiv.org/pdf/1703.05407.pdf)). The idea is simple, but extremely creative. Have the learner simulate a game between two (internal) agents in the environment it later needs to perform a task in. In this game, agent A first interacts in some random way with the environment, and agent B is then tasked with reaching the final state of agent A in as little time as possible. Agent A gradually increases the difficulty of her action (this is achieved by rewarding A when it takes simple actions that take B longer to imitate). In this manner, B learns to achieve (arbitrary) tasks and interact with the environment without any external feedback from an oracle. This idea is super-cool because it addresses one of the fundamental drawbacks of current RL systems (data-efficiency) by taking an unsupervised pre-training of an RL agent, and there are some promising preliminary results. I imagine (hope) that there will quite a bit of follow-up to these ideas from the RL community.

## [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/)

Finally, as the keynote speaker of the event, Zoubin gave an overview of the main current challenges of AI as he sees them (this was one of the main themes of the forum). He mentioned a number of important challenges, but the overall theme of the talk was the importance of addressing uncertainty. Looking forward, it is crucial to build AI systems that know what they don't know, and can reason and act taking this into account. ML (and especially DL) systems are often overconfident of their predictions, even when in practice they are incorrect. This is acceptable for one-off recognition tasks such as object detection, but when these predictions are fed to downstream, real-world tasks (e.g., self-driving cars), over-confident predictions can have unacceptable outcomes. Uncertainty can potentially help with data-efficiency, another major drawback of current ML models. The need for datasets containing millions of labeled examples to perform recognition tasks (or trials for RL agents) bottlenecks advances in many tasks/domains where massive datasets are unavailable or infeasible to generate, and uncertainty is often key to reduce these requirements. I appreciated Zoubin's talk very much because it is easy to get carried away with the hype around ML/DL, and it is important to remember that there immense and important challenges that we still face.

To summarize, I greatly enjoyed this experience. It is great to see the level of industry involvement and commitment to research in our field first hand, and I was very happy to get to know some of my colleagues from other universities. I hope that the collaboration with Samsung is fruitful, and I eagerly await the next AI forum.  


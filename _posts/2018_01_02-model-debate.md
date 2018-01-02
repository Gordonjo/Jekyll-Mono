---
layout: post
title: On Model-Based vs. Model-Free AI 
author: Jonathan Gordon
comments: true
---

An interesting debate has arisen lately in the machine learning community concerning two competing (?) approaches to ML and (more generally) AI. The debate is rather high-level, but in my opinion touches upon something that is at the very core of research in the field. In this post I will lay out the fundamental aspects of the debate as I see them, and try and give my personal perspective on the issue.

## The Rise of Deep Learning
----

Undoubtably the most dominant trend in ML at the moment is deep learning - i.e., learning that is based on neural networks and their offspring achieved by applying gradient-based methods (error-back propagation, a.k.a backprop). DL has led to some undeniably outstanding successes, achieving human-level (or better) performance in specific tasks such as object recognition, speech (recognition and synthesis), and game-playing. This in turn has led to wide-spread adoption and integration of DL technologies in many leading tech companies, and has generated a lot of media hype and public recognition. 

Despite these impressive successes, there are major drawbacks to DL. The obvious problems are terrible data-efficiency and an inability to generalize across tasks or domains. Both of these characteristics arise from the fact that neural networks are at their core _pattern matching_ machines. Essentially, they find complex patterns in massive datasets that correlate with a desired output. Often, these patterns may correspond to interesting notions of underlying structure in the data (i.e., edges in an image), lending to the notion of _representation learning_. However, these representations are highly specified for specific tasks, and training a network for a new task must be done from scratch. This can be frustrating since we often _intuitively know_ there must be transferable knowledge between the tasks. An example of this is the phenomenon known as catastrophic forgetting [[^1]], where performance of a trained network on task A will deteriorate significantly when trained for related task B. Essentially, every time we train a neural network we must do so (pretty much) from scratch, even if there are many shared components that _should_ be leveraged.


## A Real-World Counter Example
----

Given the recent successes of DL and gradient-based learning, and the impressive ability of generic neural networks to learn meaningful representations, should we in fact consider alternative approaches? Well, we have a biological counter-example of general intelligence that works - humans. As many have pointed out, artificial flight was achieved when humans moved away from biological inspirations. This is a valid point, and I do not believe we should limit our investigation of intelligent systems to mimicking human intelligence. On the other hand, human intelligence provides an excellent benchmark for measuring performance of artificial systems _and_ a source from which to draw aspirations.

In context of this post, we observe that humans learn rich representations that generalize across complex tasks from very few examples. For instance, given a single visual example of a new object, humans easily infer high level traits such as its purpose, decomposition into parts and the relations between them, and how it may interact with different environments (image and motivation taken from [[^2]]). They can then use these traits to classify new visual examples of the object, draw (or imagine) variations on the object, or think of uses for it. All of this from observing a single example.

{:refdef: style="text-align: center;"}
<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/human_concepts.png" width="100%" height="100%">
{:refdef}

The presence of these traits in human intelligence serves both to highlight the drawbacks of neural-network based learning and perhaps indicate that DL alone will not be sufficient to achieve significant progress towards more general intelligence. Further, (though I am no neuroscientist), there seems to be sufficient evidence it is highly unlikely the brain is composed of sets of generic, single purpose neural networks [[^3]]. 

In [[^3]], the authors argue that the ability of humans to learn such complex representations of new concepts given little data is due to the existence of _models_ that allow humans to leverage their past experiences to do so. They go on to argue that important characteristics of these models are the presence of _fundamental learning structures_ from early age (e.g., intuitive physical and psychological knowledge), _causality_, and _compositionality_. Importantly, none of these are characteristics of DL.


### The Heart of the Debate

Given what is discussed above, we can now ask the central question, which is exemplified in two recent papers [[^3], [^4]]: how much emphasis should we place on encoding knowledge into our models?

## The Case for Model-Free ML
----

Among the main proponents of _learning from scratch_ and largely _model free_ ML are researchers at Google DeepMind, who authored [[^4]]. DeepMind are responsible for some of the most noteworthy successes of DL such as mastering Atari games [[^5]] and Go [[^6]]. These successes have been achieved with advances in model-free systems. Indeed, the only inductive biases introduced in these systems is that of standard convolutional nets for parsing images. As discussed in [[^4]], there are (at least) two significant arguments against incoporating and encoding prior knowledge into our models.

The first has to do with the notion of generality (_not_ generalization). For many domains, prior or expert knowledge may not be available, or may be intractable to encode. For instance, physical laws can provide important prior knowledge in domains such as robotic movement, and the model-based approach justifies leveraging that knowledge. This may indeed improve performance of deployed systems. However, in many domains (e.g., healthcare, dialogue systems) such knowledge may not be so straight-forward to include. How would a general model-based approach towards intelligence then deal with these domains? To quote [[^4]]: 

"_... it is not clear that detailed knowledge engineering will be realistically attainable in all areas we will want our agents to tackle_". 

With this in mind, generic models that make no prior assumptions on the domain and "learn from scratch" may be more generally applicable to a wider range of areas.

The second point has to do with avoiding encoding our own biases into intelligent agents. The knowledge and inductive biases that [[^3]] argue should be included in our intelligent models are specifically human. There is no reason to believe that these principles are required (let alone _optimal_) for machines to be intelligent. Indeed, encoding incomplete notions of inductive biases in an attempt to mimick the human brain may _hinder_ progress, much as attempting to achieve artificial flight by mimicking birds proved unproductive. Proponents of model-free learning argue that by starting from a 'blank slate', machines generate representations and inductive biases that are useful to them for the specified task. From this view, learning from scratch is a feature rather than a bug.  

Proponents of the model free approach argue that in fact we should avoid encoding knowledge into our models for the reasons specified above. Existing problems in neural networks, such as data efficiency and transferability, can be solved within the context of DL and do not require prior/expert knowledge. Recent advances in DL (such as memory or attention mechanisms, deep generative models) are indeed steps in these directions, though it is not clear that these will solve the core issues.



## How Much is Really Gained by Models?
----

The main motivation for heavily engineered models is achieving human-like learning: rich, generalizable representations of complex concepts with little examples. However, is it clear that encoding knowledge into models can actually achieve this? Below I discuss an example that shows how heavily engineered models can indeed get around some of the major difficulties of DL.

### Game-Playing with Minor Environmental Variations

In [[^5]], the authors show that a single deep RL algorithm can achieve human-level performance on a wide range of Atari games using only the screen pixels as input. This is a very impressive achievement, and one that has been subsequently improved upon. However, in [[^7]] the authors point out a major flaw of those deep RL agents: minor variations in the environment completely break performance. 

{:refdef: style="text-align: center;"}
<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/breakout.png" width="100%" height="100%">
{:refdef}

In other words, the "knowledge" gained by the agent while learning to play the original version of the game is so specific that any variation to the environment (including ones that humans easily adapt to) is a completely new domain and the agent must be retrained. Clearly, this is unsatisfying, and we desire our agents to be more general than this.

The authors go on to propose schema networks; complex graphical models that model objects, their movements and attributes, and causal relations between actions/objects/reward. This allows the agent to perform long term planning (phrased as inference) after training. The paper then details experiments showing that schema networks easily adapt to the variations in environment that break the deep RL agents. 

Beside causal relations, the schema network model encodes knowledge on intuitive physics, namely that objects are smooth, and contain physical attributes relating to movement, size, etc. This paper demonstrates how encoding fundamental principles of causality and physics can enable models to be more robust to changes (as well as learn more efficiently).


For interested readers, other excellent examples of how models improve both data-efficiency and generalization can be found in [[^2], [^8]].

## My Own Two Cents
----

Personally, I find this debate fascinating, and at the very core of what AI and ML research are all about. Both sides make compelling points, and it is not clear to me that one is correct. My natural inclination leans more towards the modeling perspective: I believe that intelligence has much to do with (probabilistic) models, and whether or not these draw inspitation from human intelligence, I cannot imagine an intelligent agent that is completely model free. 

However, I see no reason for the two approaches to be mutually exclusive (this notion is also emphasized in [[^3]]). Much of the most interesting recent work has been on teaching neural networks to perform probabilistic inference. This is where I believe the most interesting and promising work currently lies: bridging the gap between DL and model-based ML. Neural networks are incredibly powerful tools, and integrating them into the toolbox of probabilistic modelling holds enormous potential. 

Further, a potential goal is to merge model-based and model-free ML. My opinion is that, where possible, we should encode (fundamental) notions such as physics and theory of mind into our models, and neural networks can provide flexible mappings for complex relations. Where no such expert knowledge exists, latent variable models with neural network parameterizations can provide a powerful avenue to allow systems to learn abstract concepts from scratch. Ideally, these notions can co-exist within single systems.

As in many debates, my hunch is that the most progress can be made by integrating both sides, trying to take the best of both worlds. 

 


## References
-----

[^1]: Goodfellow J., Ian, et al. An Emprical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks. 2013
[^2]: Lake M., Brenden, Salakhutdinov, Ruslan, and Tenenbaum B. Joshua. Human-level Concept Learning Through Probabilistic Program Induction. 2015
[^3]: Lake M., Brenden, et al. Building Machines that Learn and Think Like People. 2017
[^4]: Botvinick, Matthew, et al. Building Machines that Learn and Think for Themselves: Commentary on Lake, Ullman, Tenebaum, and Gershamn. 2017
[^5]: Mnih, Volodymyr, et al. Human-Level Control Through Deep Reinforcement Learning. 2015 
[^6]: Silver, David, et al. Mastering the Game of Go Without Human Knowledge. 2017
[^7]: Kansky, Ken, et al. Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics. 2017
[^8]: George, Dileep, et al. A Generative Vision Model that Trains with High Data-Efficiency and Breaks Text-Based CAPTCHAs. 2017

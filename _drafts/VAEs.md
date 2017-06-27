---
layout: post
title: Training Deep Models with Stochastic Backpropagation (Part 2 - Variational Autoencoder)
author: Jonathan Gordon
---

In my previous post, I set up some background for training deep generative models with stochastic backpropagation. In this post what I want to do is walk through a tensorflow implementation of a variational autoencoder (VAE) with some toy data. I specifically chose data that highlights some possible pitfalls of the VAE, and I'll show how we might diagnose and deal with these. 

## Variational Autoencoder
-----

Before we dive in, let's briefly just review what a VAE is. We can partially describe a VAE with the following graphical model:

{:refdef: style="text-align: center;"}
<img src="https://raw.githubusercontent.com/Gordonjo/Jekyll-Mono/gh-pages/images/vae.png" width="20%" height="20%">
{:refdef}

Here, \\(z\\) is a latent variable, \\(x\\) is out input, and \\(\theta\\) parameterizes the conditional distribution of \\(x\\) given \\(z\\). Specific choices for the generative model and inference network are what specify the VAE. Specifically, we can specify:

\begin{equation}
p(z) = \mathcal{N}(z; 0,1); p(x|z) = f_{\theta}(x; z, \theta)
\end{equation}

with \\(f\\) being a valid distribution (usually Gaussian or Bernoulli for continuous and binary \\(x\\) respectively) whose parameters are outputs of neural networks. Recalling my previous post, we also need to specify an inference network. For instance, we can choose:

\begin{equation}
q_{\phi}(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x))
\end{equation}

where again, \\(\mu(x)\\) and \\(\sigma(x)\\) are parameterized by neural networks. Usually, we will use a diagonal covariance matrix for \\(q\\). This is a strong assumption and is a point of major dispute and active research, but in practice this works well for simple cases. This leads to a standard normal distribution for \\(\epsilon\\) (reparameterization), and:

\begin{equation}
g_{\phi}(\epsilon, x) = \mu(x) + \epsilon \otimes \sigma(x)
\end{equation}

VAEs are popular because they represent a princpled approach to performing deep unsupervised learning. \\(z\\) can be thought of as a latent, low-dimensional encoding of \\(x\\), and has been shown to capture nice intuitive characterstics of inputs. For instance, if \\(x\\) is a set of facial images from a single person, \\(z\\) has been known to capture things like rotation of the face, some notion of facial expression, etc'. Another nice things about VAEs is that they can generate synthetic examples that mimic the original data in a compelling manner. 

## Optimization and Training
-----

The mathematics are pretty mechanic, and we can train the model and inference network jointly using stochastic backpropagation. Just to recap, the lower bound (ELBO) that we will maximize can be expressed as:

\begin{equation}
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q(z|x)}[\log p(x|z) + \log p(z) - \log q(z|x)]
\end{equation}

Here I have broken down the KL term into an expectation of log probabilities. This is a more general form that can be applied to different priors and inference networks, and needs to be sampled rather than evaluated analytically. I tend to use this form in my implementations for generality. The approximation of the objective (to minimize) can be expressed as:

\begin{equation}
\mathcal{J}(\theta, \phi; x) = -\frac{1}{L} \sum\limits_{l=1}^L \log p(x|z^l) + \log p(z^l) - \log q(z^l|x)
\end{equation}

where we are sampling \\(z^l \sim q\\) with the reparameterization trick. In practice, \\(L=1\\) is typically enough, and converges nicely. 

## TensorFlow Implementation 
-----

One of the great things about VAEs is how easy they are to implement. Lately I've been working in TensorFlow, which is very convenient for this. Let's step through the key aspects of the implementation (for a complete code visit my GitHub page). The first thing to do is set up the necessary placeholders:

{% highlight python %}
def _create_placeholders(self):
    self.x_batch = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.X_DIM], name='x_batch')
    self.x_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.X_DIM], name='x_train')
    self.x_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.X_DIM], name='x_test')
{% endhighlight %}

Next, we initialize the generative and inference networks. Here, I am calling upon a library I have constructed to handle neural networks parameterizing distributions. Each network is a dictionary of weights and biases, and the supporting library handles forward passes through the network.

{% highlight python %}
def _initialize_networks(self):
    if self.TYPE_PX=='Gaussian':
        self.Pz_x = dgm._init_Gauss_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')
    elif self.TYPE_PX=='Bernoulli':
        self.Pz_x = dgm._init_Cat_net(self.Z_DIM, self.NUM_HIDDEN, self.X_DIM, 'Pz_x_')	   
    self.Qx_z = dgm._init_Gauss_net(self.X_DIM, self.NUM_HIDDEN, self.Z_DIM, 'Qx_z_')
{% endhighlight %}

Next, we can define the ELBO bound computation. The helper library handles (diagonal covariance) Gaussian log-likelihoods; the function expects the data evaluated, mean and log-variance of the distribution. I have placed that as the first function.

{% highlight python %}
def _gauss_logp(x, mu, log_var):
    b_size = tf.cast(tf.shape(mu)[0], tf.float32)
    D = tf.cast(tf.shape(x)[1], tf.float32)
    xc = x - mu
    return -0.5*(tf.reduce_sum((xc * xc) / tf.exp(log_var), axis=1) + 
    				tf.reduce_sum(log_var, axis=1) + D * tf.log(2.0*np.pi))

def _sample_Z(self, x, n_samples=1):
	mean, log_var = dgm._forward_pass_Gauss(x, self.Qx_z, self.NUM_HIDDEN, self.NONLINEARITY)
	eps = tf.random_normal([tf.shape(x)[0], self.Z_DIM], dtype=tf.float32)
	return mean, log_var, mean + tf.sqrt(tf.exp(log_var)) * eps 

def _compute_logpx(self, x, z):
    if self.TYPE_PX == 'Gaussian':
        mean, log_var = dgm._forward_pass_Gauss(z,self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
        return dgm._gauss_logp(x, mean, log_var)
    elif self.TYPE_PX == 'Bernoulli':
        logits = dgm._forward_pass_Cat_logits(z, self.Pz_x, self.NUM_HIDDEN, self.NONLINEARITY)
        return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits),axis=1)

 def _compute_ELBO(self, x):
    z_mean, z_log_var, z = self._sample_Z(x)
    l_qz = dgm._gauss_logp(z, z_mean, z_log_var)
    l_pz = dgm._gauss_logp(z, tf.zeros_like(z), tf.ones_like(z)) 
    l_px = self._compute_logpx(x, z)
    return tf.reduce_mean(l_px + l_pz - l_qz)
{% endhighlight %}

Thats it. All that's left is to implement a fit method. The class here interfaces with a data class that I have constructed for my more general research purposes, and fit here assumes the data as such. The class yields mini-batches according to different schemes, but here I use standard minibatching.

{% highlight python %}
def fit(self, Data, n_epochs):
    self._create_placeholders()
    self._initialize_networks()
    self.loss = -self._compute_ELBO(self.x_batch)
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        while epoch < self.NUM_EPOCHS:
            x_batch, _ = Data.next_batch_regular(self.BATCH_SIZE)
            _, loss_batch = sess.run([self.optimizer, self.loss], 
                                      feed_dict={self.x_batch:x_batch})
            if Data._epochs > epoch:
                print('Epoch: {}, ELBO: {:5.3f},'.format(epoch,loss_batch)) 
{% endhighlight %}

That's all there is to it. Of course, these are only the core lines of code, and this is enough to run the model. The rest is fluff and filler to allow multiple datasets, plotting options, logging with TF etc'. You can see all of this if you are interested on the GitHub page.
# DENOISING DIFFUSION IMPLICIT MODELS

Jiaming Song, Chenlin Meng & Stefano Ermon

Stanford University

{tsong,chenlin,ermon}@cs.stanford.edu

## ABSTRACT

Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps in order to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a particular Markovian diffusion process. We generalize DDPMs via a class of non-Markovian diffusion processes that lead to the same training objective. These non-Markovian processes can correspond to generative processes that are deterministic, giving rise to implicit models that produce high quality samples much faster. We empirically demonstrate that DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, perform semantically meaningful image interpolation directly in the latent space, and reconstruct observations with very low error.

## 1 INTRODUCTION

Deep generative models have demonstrated the ability to produce high quality samples in many domains (Karras et al., 2020; van den Oord et al., 2016a). In terms of image generation, generative adversarial networks (GANs, Goodfellow et al. (2014)) currently exhibits higher sample quality than likelihood-based methods such as variational autoencoders (Kingma & Welling, 2013), autoregressive models (van den Oord et al., 2016b) and normalizing flows (Rezende & Mohamed, 2015; Dinh et al., 2016). However, GANs require very specific choices in optimization and architectures in order to stabilize training (Arjovsky et al., 2017; Gulrajani et al., 2017; Karras et al., 2018; Brock et al., 2018), and could fail to cover modes of the data distribution (Zhao et al., 2018).

Recent works on iterative generative models (Bengio et al., 2014), such as denoising diffusion probabilistic models (DDPM, Ho et al. (2020)) and noise conditional score networks (NCSN, Song & Ermon (2019)) have demonstrated the ability to produce samples comparable to that of GANs, without having to perform adversarial training. To achieve this, many denoising autoencoding models are trained to denoise samples corrupted by various levels of Gaussian noise. Samples are then produced by a Markov chain which, starting from white noise, progressively denoises it into an image. This generative Markov Chain process is either based on Langevin dynamics (Song & Ermon, 2019) or obtained by reversing a forward diffusion process that progressively turns an image into noise (Sohl-Dickstein et al., 2015).

A critical drawback of these models is that they require many iterations to produce a high quality sample. For DDPMs, this is because that the generative process (from noise to data) approximates the reverse of the forward diffusion process (from data to noise), which could have thousands of steps; iterating over all the steps is required to produce a single sample, which is much slower compared to GANs, which only needs one pass through a network. For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on a Nvidia 2080 Ti GPU. This becomes more problematic for larger images as sampling 50k images of size 256 × 256 could take nearly 1000 hours on the same GPU.

To close this efficiency gap between DDPMs and GANs, we present denoising diffusion implicit models (DDIMs). DDIMs are implicit probabilistic models (Mohamed & Lakshminarayanan, 2016) and are closely related to DDPMs, in the sense that they are trained with the same objective function.

![](images/52aed20e2bc48543879fd7457b5ba9cf86d148098834d6b445729e338a322a0d.jpg)  
Figure 1: Graphical models for diffusion (left) and non-Markovian (right) inference models.

In Section 3, we generalize the forward diffusion process used by DDPMs, which is Markovian, to non-Markovian ones, for which we are still able to design suitable reverse generative Markov chains. We show that the resulting variational training objectives have a shared surrogate objective, which is exactly the objective used to train DDPM. Therefore, we can freely choose from a large family of generative models using the same neural network simply by choosing a different, non-Markovian diffusion process (Section 4.1) and the corresponding reverse generative Markov Chain. In particular, we are able to use non-Markovian diffusion processes which lead to ”short” generative Markov chains (Section 4.2) that can be simulated in a small number of steps. This can massively increase sample efficiency only at a minor cost in sample quality.

In Section 5, we demonstrate several empirical benefits of DDIMs over DDPMs. First, DDIMs have superior sample generation quality compared to DDPMs, when we accelerate sampling by 10× to 100× using our proposed method. Second, DDIM samples have the following “consistency” property, which does not hold for DDPMs: if we start with the same initial latent variable and generate several samples with Markov chains of various lengths, these samples would have similar high-level features. Third, because of “consistency” in DDIMs, we can perform semantically meaningful image interpolation by manipulating the initial latent variable in DDIMs, unlike DDPMs which interpolates near the image space due to the stochastic generative process.

## 2 BACKGROUND

Given samples from a data distribution q(x<sub>0</sub>), we are interested in learning a model distribution p<sub>θ</sub>(x<sub>0</sub>) that approximates q(x<sub>0</sub>) and is easy to sample from. Denoising diffusion probabilistic models (DDPMs, Sohl-Dickstein et al. (2015); Ho et al. (2020)) are latent variable models of the form

![](images/b1a7575705cb97b98f42a8b044cc3e50a67c911b32fd0277b7087153849d7e77.jpg)

(1)

where x , . . . , x are latent variables in the same sample space as x (denoted as X). The parameters θ are learned to fit the data distribution q(x ) by maximizing a variational lower bound:

![](images/6b253c3a01a1475cc9bea1e8325fbc28095dceaca918b17e955c1985b09ac208.jpg)

(2)

where q(x<sub>1:T</sub>|x<sub>0</sub>) is some inference distribution over the latent variables. Unlike typical latent variable models (such as the variational autoencoder (Rezende et al., 2014)), DDPMs are learned with a fixed (rather than trainable) inference procedure q(x |x ), and latent variables are relatively high dimensional. For example, Ho et al. (2020) considered the following Markov chain with Gaussian transitions parameterized by a decreasing sequence α<sub>1:T</sub> ∈ (0, 1]<sup>T</sup>:

![](images/30758435f31a0e34ad72a8be7917a3c18eeaae02eb3ef7df85560c3679d8d2e0.jpg)

(3)

where the covariance matrix is ensured to have positive terms on its diagonal. This is called the forward process due to the autoregressive nature of the sampling procedure (from x to x ). We call the latent variable model p<sub>θ</sub>(x<sub>0:T</sub>), which is a Markov chain that samples from x<sub>T</sub> to x<sub>0</sub>, the generative process, since it approximates the intractable reverse process q(x<sub>t−1</sub>|x<sub>t</sub>). Intuitively, the forward process progressively adds noise to the observation x , whereas the generative process progressively denoises a noisy observation (Figure 1, left).

A special property of the forward process is that

![](images/f8ecd8748cff7fe5dbe8bf6739e86c102aeaa6bbd3eb4c609475fcf7ea8c3853.jpg)

so we can express x as a linear combination of x and a noise variable :

![](images/be42be09265a44f4b9080dfd8bd5a0ca61b74caeedb891d97def457e86d1f21f.jpg)

(4)

When we set α<sub>T</sub> sufficiently close to 0, q(x<sub>T</sub>|x<sub>0</sub>) converges to a standard Gaussian for all x<sub>0</sub>, so it is natural to set p<sub>θ</sub>(x<sub>T</sub>) := N (0, I). If all the conditionals are modeled as Gaussians with trainable mean functions and fixed variances, the objective in Eq. (2) can be simplified to<sup>1</sup>:

![](images/282dfc7df2516d9abc485c151757518202208f71132516ee8869ec5d9e980ed6.jpg)

(5)

where <sub>θ</sub> := {<sup>(t)</sup><sub>θ</sub> }<sup>T</sup><sub>t=1</sub> is a set of T functions, each <sup>(t)</sup><sub>θ</sub> : X → X (indexed by t) is a function with trainable parameters θ<sup>(t)</sup>, and γ := [γ<sub>1</sub>, . . . , γ<sub>T</sub>] is a vector of positive coefficients in the objective that depends on α . In Ho et al. (2020), the objective with γ = 1 is optimized instead to maximize generation performance of the trained model; this is also the same objective used in noise conditional score networks (Song & Ermon, 2019) based on score matching (Hyvarinen¨ , 2005; Vincent, 2011). From a trained model, x is sampled by first sampling x from the prior p (x ), and then sampling x<sub>t−1</sub> from the generative processes iteratively.

The length T of the forward process is an important hyperparameter in DDPMs. From a variational perspective, a large T allows the reverse process to be close to a Gaussian (Sohl-Dickstein et al., 2015), so that the generative process modeled with Gaussian conditional distributions becomes a good approximation; this motivates the choice of large T values, such as T = 1000 in Ho et al. (2020). However, as all T iterations have to be performed sequentially, instead of in parallel, to obtain a sample x<sub>0</sub>, sampling from DDPMs is much slower than sampling from other deep generative models, which makes them impractical for tasks where compute is limited and latency is critical.

## 3 VARIATIONAL INFERENCE FOR NON-MARKOVIAN FORWARD PROCESSES

Because the generative model approximates the reverse of the inference process, we need to rethink the inference process in order to reduce the number of iterations required by the generative model. Our key observation is that the DDPM objective in the form of L only depends on the marginals<sup>2</sup> q(x |x ), but not directly on the joint q(x |x ). Since there are many inference distributions (joints) with the same marginals, we explore alternative inference processes that are non-Markovian, which leads to new generative processes (Figure 1, right). These non-Markovian inference process lead to the same surrogate objective function as DDPM, as we will show below. In Appendix A, we show that the non-Markovian perspective also applies beyond the Gaussian case.

## 3.1 NON-MARKOVIAN FORWARD PROCESSES

![](images/63f7d48b1e7493803f14e76484a7f79c0b0c23e4d2e746774d38b31e21880e0f.jpg)

(6)

<sub>where</sub> <sub>qσ(xT|x0)</sub> <sub>=</sub> N <sub>(</sub>√<sub>αTx0, (1</sub> <sub>−</sub> <sub>αT)I)</sub> <sub>and</sub> <sub>for</sub> <sub>all</sub> <sub>t</sub> <sub>></sub> <sub>1,</sub>

![](images/4e418cf377cf2c66c282da0f2181fa867d66b16f7e0d45b464e0f745cd7b4b38.jpg)

(7)

<sub>The</sub> <sub>mean</sub> <sub>function</sub> <sub>is</sub> <sub>chosen</sub> <sub>to</sub> <sub>order</sub> <sub>to</sub> <sub>ensure</sub> <sub>that</sub> <sub>qσ(xt|x0)</sub> <sub>=</sub> N <sub>(</sub>√<sub>αtx0, (1</sub> <sub>−</sub> <sub>αt)I)</sub> <sub>for</sub> <sub>all</sub> t (see Lemma 1 of Appendix B), so that it defines a joint inference distribution that matches the “marginals” as desired. The forward process<sup>3</sup> can be derived from Bayes’ rule:

![](images/e1f7cf37bd6aad3972622c0903556ea307512bc818936229b16c850299d9eb48.jpg)

(8)

which is also Gaussian (although we do not use this fact for the remainder of this paper). Unlike the diffusion process in Eq. (3), the forward process here is no longer Markovian, since each x<sub>t</sub> could depend on both x <sub>−</sub> and x . The magnitude of σ controls the how stochastic the forward process is; when σ → 0, we reach an extreme case where as long as we observe x and x for some t, then x<sub>t−1</sub> become known and fixed.

## 3.2 GENERATIVE PROCESS AND UNIFIED VARIATIONAL INFERENCE OBJECTIVE

edge of q (x |x , x ). Intuitively, given a noisy observation x , we first make a prediction<sup>4</sup> of the corresponding x<sub>0</sub>, and then use it to obtain a sample x<sub>t−1</sub> through the reverse conditional distribution q<sub>σ</sub>(x<sub>t−1</sub>|x<sub>t</sub>, x<sub>0</sub>), which we have defined.

attempts to predict <sub>t</sub> from x<sub>t</sub>, without knowledge of x<sub>0</sub>. By rewriting Eq. (4), one can then predict the denoised observation, which is a prediction of x<sub>0</sub> given x<sub>t</sub>:

![](images/a2087faa0fd9cc6c600020e6799dcaeb169830c17ce4c950dd9ad54af60a16dd.jpg)

(9)

We can then define the generative process with a fixed prior p<sub>θ</sub>(x<sub>T</sub>) = N(0, I) and

![](images/f149234af43fd21c180fa4a1a3f0e4025ffd376ccd582fb591c062535b4025ca.jpg)

(10)

where q<sub>σ</sub>(x<sub>t−1</sub>|x<sub>t</sub>, f<sup>(t)</sup>(x<sub>t</sub>)) is defined as in Eq. (7) with x<sub>0</sub> replaced by f<sup>(t)</sup>(x<sub>t</sub>). We add some Gaussian noise (with covariance σ<sup>2</sup>I) for the case of t = 1 to ensure that the generative process is supported everywhere.

We optimize θ via the following variational inference objective (which is a functional over <sub>θ</sub>):

![](images/493532adc09f8d448aa17679acf9d29f9ad6fbdd4c3e490a81e47e23711e7c41.jpg)

where we factorize q<sub>σ</sub>(x<sub>1:T</sub>|x<sub>0</sub>) according to Eq. (6) and p<sub>θ</sub>(x<sub>0:T</sub>) according to Eq. (1).

From the definition of J , it would appear that a different model has to be trained for every choice of σ, since it corresponds to a different variational objective (and a different generative process). However, J is equivalent to L for certain weights γ, as we show below.

Theorem 1. For all σ > 0, there exists γ ∈ R<sup>T</sup> and C ∈ R, such that J<sub>σ</sub> = L<sub>γ</sub> + C.

The variational objective L<sub>γ</sub> is special in the sense that if parameters θ of the models  (t) are not shared across different t, then the optimal solution for  will not depend on the weights γ (as global optimum is achieved by separately maximizing each term in the sum). This property of L<sub>γ</sub> has two implications. On the one hand, this justified the use of L<sub>1</sub> as a surrogate objective function for the variational lower bound in DDPMs; on the other hand, since J<sub>σ</sub> is equivalent to some L<sub>γ</sub> from Theorem 1, the optimal solution of J<sub>σ</sub> is also the same as that of L<sub>1</sub>. Therefore, if parameters are not shared across t in the model  , then the L objective used by Ho et al. (2020) can be used as a surrogate objective for the variational objective J<sub>σ</sub> as well.

## 4 SAMPLING FROM GENERALIZED GENERATIVE PROCESSES

With L<sub>1</sub> as the objective, we are not only learning a generative process for the Markovian inference process considered in Sohl-Dickstein et al. (2015) and Ho et al. (2020), but also generative processes for many non-Markovian forward processes parametrized by σ that we have described. Therefore, we can essentially use pretrained DDPM models as the solutions to the new objectives, and focus on finding a generative process that is better at producing samples subject to our needs by changing σ.

![](images/7d720719b43b279c954ab23a950ede366a6d44a8ec0129c630b1f19bc7975d0d.jpg)  
Figure 2: Graphical model for accelerated generation, where τ = [1, 3].

## 4.1 DENOISING DIFFUSION IMPLICIT MODELS

From p<sub>θ</sub>(x<sub>1:T</sub>) in Eq. (10), one can generate a sample x<sub>t−1</sub> from a sample x<sub>t</sub> via:

![](images/f1d8bce3529beca443dcd4708176d1c7121a030d40fa6705ecaae423ca409cef.jpg)

(12)

where  ∼ N(0, I) is standard Gaussian noise independent of x , and we define α := 1. Different choices of σ values results in different generative processes, all while using the same model  , so re-training the model is unnecessary. When σ<sub>t</sub> = p(1 − α<sub>t−1</sub>)/(1 − α<sub>t</sub>)p1 − α<sub>t</sub>/α<sub>t−1</sub> for all t, the forward process becomes Markovian, and the generative process becomes a DDPM.

We note another special case when σ = 0 for all t<sup>5</sup>; the forward process becomes deterministic given x and x , except for t = 1; in the generative process, the coefficient before the random noise  becomes zero. The resulting model becomes an implicit probabilistic model (Mohamed & Lakshminarayanan, 2016), where samples are generated from latent variables with a fixed procedure (from x<sub>T</sub> to x<sub>0</sub>). We name this the denoising diffusion implicit model (DDIM, pronounced /d:Im/), because it is an implicit probabilistic model trained with the DDPM objective (despite the forward process no longer being a diffusion).

## 4.2 ACCELERATED GENERATION PROCESSES

In the previous sections, the generative process is considered as the approximation to the reverse process; since of the forward process has T steps, the generative process is also forced to sample T steps. However, as the denoising objective L<sub>1</sub> does not depend on the specific forward procedure as long as q<sub>σ</sub>(x<sub>t</sub>|x<sub>0</sub>) is fixed, we may also consider forward processes with lengths smaller than T, which accelerates the corresponding generative processes without having to train a different model.

Let us consider the forward process as defined not on all the latent variables x<sub>1:T</sub>, but on a subset {x<sub>τ</sub> , . . . , x<sub>τ</sub> }, where τ is an increasing sub-sequence of [1, . . . , T] of length S. In particular, we define the sequential forward process over x<sub>τ</sub> , . . . , x<sub>τ</sub> such that q(x<sub>τ</sub> |x<sub>0</sub>) = N (√α<sub>τ</sub> x<sub>0</sub>, (1 − α<sub>τ</sub> )I) <sub>matches</sub> <sub>the</sub> <sub>“marginals”</sub> <sub>(see</sub> <sub>Figure</sub> <sub>2</sub> <sub>for</sub> <sub>an</sub> <sub>illustration).</sub> <sub>The</sub> <sub>generative</sub> process now samples latent variables according to reversed(τ), which we term (sampling) trajectory. When the length of the sampling trajectory is much smaller than T, we may achieve significant increases in computational efficiency due to the iterative nature of the sampling process.

Using a similar argument as in Section 3, we can justify using the model trained with the L objective, so no changes are needed in training. We show that only slight changes to the updates in Eq. (12) are needed to obtain the new, faster generative processes, which applies to DDPM, DDIM, as well as all generative processes considered in Eq. (10). We include these details in Appendix C.1.

In principle, this means that we can train a model with an arbitrary number of forward steps but only sample from some of them in the generative process. Therefore, the trained model could consider many more steps than what is considered in (Ho et al., 2020) or even a continuous time variable t (Chen et al., 2020). We leave empirical investigations of this aspect as future work.

## 4.3 RELEVANCE TO NEURAL ODES

Moreover, we can rewrite the DDIM iterate according to Eq. (12), and its similarity to Euler integration for solving ordinary differential equations (ODEs) becomes more apparent:

![](images/bf4e195f2eb0c93c38d3b2bafe98d6ab262f412e43d6edee0c3ed524c4be19e0.jpg)

(13)

<sub>To derive the corresponding ODE, we can reparameterize (</sub>√<sub>1 − α/</sub>√<sub>α) with σ and (x/</sub>√<sub>α) with</sub> x¯. In the continuous case, σ and x are functions of t, where σ : R<sub>≥0</sub> → R<sub>≥0</sub> is continous, increasing with σ(0) = 0. Equation (13) with can be treated as a Euler method over the following ODE:

![](images/f706e1962d8647d4d7754af7a44da41345f1f7c5a7e014293bd9aa6620d5dfbc.jpg)

(14)

where the initial conditions is x(T) ∼ N(0, σ(T)) for a very large σ(T) (which corresponds to the case of α ≈ 0). This suggests that with enough discretization steps, the we can also reverse the generation process (going from t = 0 to T), which encodes x<sub>0</sub> to x<sub>T</sub> and simulates the reverse of the ODE in Eq. (14). This suggests that unlike DDPM, we can use DDIM to obtain encodings of the observations (as the form of x<sub>T</sub>), which might be useful for other downstream applications that requires latent representations of a model.

In a concurrent work, (Song et al., 2020) proposed a “probability flow ODE” that aims to recover the marginal densities of a stochastic differential equation (SDE) based on scores, from which a similar sampling schedule can be obtained. Here, we state that the our ODE is equivalent to a special case of theirs (which corresponds to a continuous-time analog of DDPM).

Proposition 1. The ODE in Eq. (14) with the optimal model <sup>(t)</sup><sub>θ</sub> has an equivalent probability flow ODE corresponding to the “Variance-Exploding” SDE in Song et al. (2020).

We include the proof in Appendix B. While the ODEs are equivalent, the sampling procedures are not, since the Euler method for the probability flow ODE will make the following update:

![](images/7642e7dfdc63c618f8671f4c55d44debc3fabc095d3f01d9baf27018e902e508.jpg)

(15)

which is equivalent to ours if α<sub>t</sub> and α<sub>t−∆t</sub> are close enough. In fewer sampling steps, however, these choices will make a difference; we take Euler steps with respect to dσ(t) (which depends less directly on the scaling of “time” t) whereas Song et al. (2020) take Euler steps with respect to dt.

## 5 EXPERIMENTS

In this section, we show that DDIMs outperform DDPMs in terms of image generation when fewer iterations are considered, giving speed ups of 10× to 100× over the original DDPM generation process. Moreover, unlike DDPMs, once the initial latent variables x are fixed, DDIMs retain highlevel image features regardless of the generation trajectory, so they are able to perform interpolation directly from the latent space. DDIMs can also be used to encode samples that reconstruct them from the latent code, which DDPMs cannot do due to the stochastic sampling process.

For each dataset, we use the same trained model with T = 1000 and the objective being L from Eq. (5) with γ = 1; as we argued in Section 3, no changes are needed with regards to the training procedure. The only changes that we make is how we produce samples from the model; we achieve this by controlling τ (which controls how fast the samples are obtained) and σ (which interpolates between the deterministic DDIM and the stochastic DDPM).

We consider different sub-sequences τ of [1, . . . , T] and different variance hyperparameters σ indexed by elements of τ . To simplify comparisons, we consider σ with the form:

![](images/d63535d4e224264d4b50661de28b0b5c08d2102f68c5e28bee57eeed5c9e877b.jpg)

(16)

where η ∈ R is a hyperparameter that we can directly control. This includes an original DDPM generative process when η = 1 and DDIM when η = 0. We also consider DDPM where the random noise has a larger standard deviation than σ(1), which we denote as σˆ: σˆ<sub>τ</sub> = p1 − α<sub>τ</sub> /α<sub>τ</sub> . This is used by the implementation in Ho et al. (2020) only to obtain the CIFAR10 samples, but not samples of the other datasets. We include more details in Appendix D.

Table 1: CIFAR10 and CelebA image generation measured in FID. η = 1.0 and σˆ are cases of DDPM (although Ho et al. (2020) only considered T = 1000 steps, and S < T can be seen as simulating DDPMs trained with S steps), and η = 0.0 indicates DDIM.  
![](images/50e8ac9142a075e7d85f6c14fb507ec3985ee22187de00352d46938a9c95c577.jpg)

![](images/dc7a63909e8a17be8949ac0a770cb77063c7fb124d7c59855e1c62092900fb28.jpg)  
Figure 3: CIFAR10 and CelebA samples with dim(τ ) = 10 and dim(τ ) = 100.

## 5.1 SAMPLE QUALITY AND EFFICIENCY

In Table 1, we report the quality of the generated samples with models trained on CIFAR10 and CelebA, as measured by Frechet Inception Distance (FID (Heusel et al., 2017)), where we vary the number of timesteps used to generate a sample (dim(τ)) and the stochasticity of the process (η). As expected, the sample quality becomes higher as we increase dim(τ ), presenting a tradeoff between sample quality and computational costs. We observe that DDIM (η = 0) achieves the best sample quality when dim(τ ) is small, and DDPM (η = 1 and σˆ) typically has worse sample quality compared to its less stochastic counterparts with the same dim(τ), except for the case for dim(τ) = 1000 and σˆ reported by Ho et al. (2020) where DDIM is marginally worse. However, the sample quality of σˆ becomes much worse for smaller dim(τ), which suggests that it is ill-suited for shorter trajectories. DDIM, on the other hand, achieves high sample quality much more consistently.

In Figure 3, we show CIFAR10 and CelebA samples with the same number of sampling steps and varying σ. For the DDPM, the sample quality deteriorates rapidly when the sampling trajectory has 10 steps. For the case of σˆ, the generated images seem to have more noisy perturbations under short trajectories; this explains why the FID scores are much worse than other methods, as FID is very sensitive to such perturbations (as discussed in Jolicoeur-Martineau et al. (2020)).

In Figure 4, we show that the amount of time needed to produce a sample scales linearly with the length of the sample trajectory. This suggests that DDIM is useful for producing samples more efficiently, as samples can be generated in much fewer steps. Notably, DDIM is able to produce samples with quality comparable to 1000 step models within 20 to 100 steps, which is a 10× to 50× speed up compared to the original DDPM. Even though DDPM could also achieve reasonable sample quality with 100× steps, DDIM requires much fewer steps to achieve this; on CelebA, the FID score of the 100 step DDPM is similar to that of the 20 step DDIM.

## 5.2 SAMPLE CONSISTENCY IN DDIMS

For DDIM, the generative process is deterministic, and x<sub>0</sub> would depend only on the initial state x<sub>T</sub>. In Figure 5, we observe the generated images under different generative trajectories (i.e. different τ) while starting with the same initial x . Interestingly, for the generated images with the same initial x , most high-level features are similar, regardless of the generative trajectory. In many cases, samples generated with only 20 steps are already very similar to ones generated with 1000 steps in terms of high-level features, with only minor differences in details. Therefore, it would appear that x alone would be an informative latent encoding of the image; and minor details that affects sample quality are encoded in the parameters, as longer sample trajectories gives better quality samples but do not significantly affect the high-level features. We show more samples in Appendix D.4.

![](images/f1c00a2c6ebeedda47e537113df7e4e2e045aebe692617e8e01dd013bd748881.jpg)

![](images/18647ff656c0e6112de3799797988a20d9ac066c4c346d3652793094d388a1a7.jpg)  
Figure 4: Hours to sample 50k images with one Nvidia 2080 Ti GPU and samples at different steps.

![](images/ed65d3ff94ff6deec98405ece564fdd8beacf8891d8b93276a41ce91f5ede1cf.jpg)  
Figure 5: Samples from DDIM with the same random x<sub>T</sub> and different number of steps.

## 5.3 INTERPOLATION IN DETERMINISTIC GENERATIVE PROCESSES

![](images/20d81cfa24f8db9f868a4c3d838d13c5f5fd07d9d166a89d0df3691c8ed5df33.jpg)  
Figure 6: Interpolation of samples from DDIM with dim(τ) = 50.

Since the high level features of the DDIM sample is encoded by x , we are interested to see whether it would exhibit the semantic interpolation effect similar to that observed in other implicit probabilistic models, such as GANs (Goodfellow et al., 2014). This is different from the interpolation procedure in Ho et al. (2020), since in DDPM the same x would lead to highly diverse x due to the stochastic generative process<sup>6</sup>. In Figure 6, we show that simple interpolations in x can lead to semantically meaningful interpolations between two samples. We include more details and samples in Appendix D.5. This allows DDIM to control the generated images on a high level directly through the latent variables, which DDPMs cannot.

Table 2: Reconstruction error with DDIM on CIFAR-10 test set, rounded to 10<sup>−4</sup>.  
![](images/449db66555d0b6353621bfeb8d63edac88a1fbd6d13b8eef6ea81804a22115db.jpg)

## 5.4 RECONSTRUCTION FROM LATENT SPACE

As DDIM is the Euler integration for a particular ODE, it would be interesting to see whether it can encode from x to x (reverse of Eq. (14)) and reconstruct x from the resulting x (forward of Eq. (14))<sup>7</sup>. We consider encoding and decoding on the CIFAR-10 test set with the CIFAR-10 model with S steps for both encoding and decoding; we report the per-dimension mean squared error (scaled to [0, 1]) in Table 2. Our results show that DDIMs have lower reconstruction error for larger S values and have properties similar to Neural ODEs and normalizing flows. The same cannot be said for DDPMs due to their stochastic nature.

## 6 RELATED WORK

Our work is based on a large family of existing methods on learning generative models as transition operators of Markov chains (Sohl-Dickstein et al., 2015; Bengio et al., 2014; Salimans et al., 2014; Song et al., 2017; Goyal et al., 2017; Levy et al., 2017). Among them, denoising diffusion probabilistic models (DDPMs, Ho et al. (2020)) and noise conditional score networks (NCSN, Song & Ermon (2019; 2020)) have recently achieved high sample quality comparable to GANs (Brock et al., 2018; Karras et al., 2018). DDPMs optimize a variational lower bound to the log-likelihood, whereas NCSNs optimize the score matching objective (Hyvarinen ¨ , 2005) over a nonparametric Parzen density estimator of the data (Vincent, 2011; Raphan & Simoncelli, 2011).

Despite their different motivations, DDPMs and NCSNs are closely related. Both use a denoising autoencoder objective for many noise levels, and both use a procedure similar to Langevin dynamics to produce samples (Neal et al., 2011). Since Langevin dynamics is a discretization of a gradient flow (Jordan et al., 1998), both DDPM and NCSN require many steps to achieve good sample quality. This aligns with the observation that DDPM and existing NCSN methods have trouble generating high-quality samples in a few iterations.

DDIM, on the other hand, is an implicit generative model (Mohamed & Lakshminarayanan, 2016) where samples are uniquely determined from the latent variables. Hence, DDIM has certain properties that resemble GANs (Goodfellow et al., 2014) and invertible flows (Dinh et al., 2016), such as the ability to produce semantically meaningful interpolations. We derive DDIM from a purely variational perspective, where the restrictions of Langevin dynamics are not relevant; this could partially explain why we are able to observe superior sample quality compared to DDPM under fewer iterations. The sampling procedure of DDIM is also reminiscent of neural networks with continuous depth (Chen et al., 2018; Grathwohl et al., 2018), since the samples it produces from the same latent variable have similar high-level visual features, regardless of the specific sample trajectory.

## 7 DISCUSSION

We have presented DDIMs – an implicit generative model trained with denoising auto-encoding / score matching objectives – from a purely variational perspective. DDIM is able to generate highquality samples much more efficiently than existing DDPMs and NCSNs, with the ability to perform meaningful interpolations from the latent space. The non-Markovian forward process presented here seems to suggest continuous forward processes other than Gaussian (which cannot be done in the original diffusion framework, since Gaussian is the only stable distribution with finite variance). We also demonstrated a discrete case with a multinomial forward process in Appendix A, and it would be interesting to investigate similar alternatives for other combinatorial structures.

Moreover, since the sampling procedure of DDIMs is similar to that of an neural ODE, it would be interesting to see if methods that decrease the discretization error in ODEs, including multistep methods such as Adams-Bashforth (Butcher & Goodwin, 2008), could be helpful for further improving sample quality in fewer steps (Queiruga et al., 2020). It is also relevant to investigate whether DDIMs exhibit other properties of existing implicit models (Bau et al., 2019).

## ACKNOWLEDGEMENTS

The authors would like to thank Yang Song and Shengjia Zhao for helpful discussions over the ideas, Kuno Kim for reviewing an earlier draft of the paper, and Sharvil Nanavati and Sophie Liu for identifying typos. This research was supported by NSF (#1651565, #1522054, #1733686), ONR (N00014-19-1-2145), AFOSR (FA9550-19-1-0024), and Amazon AWS.

## REFERENCES

Martin Arjovsky, Soumith Chintala, and Leon Bottou. Wasserstein GAN. ´ arXiv preprint arXiv:1701.07875, January 2017.

David Bau, Jun-Yan Zhu, Jonas Wulff, William Peebles, Hendrik Strobelt, Bolei Zhou, and Antonio Torralba. Seeing what a gan cannot generate. In Proceedings of the IEEE International Conference on Computer Vision, pp. 4502–4511, 2019.

Yoshua Bengio, Eric Laufer, Guillaume Alain, and Jason Yosinski. Deep generative stochastic networks trainable by backprop. In International Conference on Machine Learning, pp. 226–234, January 2014.

Christopher M Bishop. Pattern recognition and machine learning. springer, 2006.

Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, September 2018.

John Charles Butcher and Nicolette Goodwin. Numerical methods for ordinary differential equations, volume 2. Wiley Online Library, 2008.

Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. WaveGrad: Estimating gradients for waveform generation. arXiv preprint arXiv:2009.00713, September 2020.

Ricky T Q Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. arXiv preprint arXiv:1806.07366, June 2018.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. arXiv preprint arXiv:1605.08803, May 2016.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672–2680, 2014.

Anirudh Goyal, Nan Rosemary Ke, Surya Ganguli, and Yoshua Bengio. Variational walkback: Learning a transition operator as a stochastic recurrent net. In Advances in Neural Information Processing Systems, pp. 4392–4402, 2017.

Will Grathwohl, Ricky T Q Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. FFJORD: Free-form continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367, October 2018.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems, pp. 5769–5779, 2017.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two Time-Scale update rule converge to a local nash equilibrium. arXiv preprint arXiv:1706.08500, June 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239, June 2020.

Aapo Hyvarinen. Estimation of Non-Normalized statistical models by score matching.¨ Journal of Machine Learning Researc h, 6:695–709, 2005.

Alexia Jolicoeur-Martineau, Remi Pich´ e-Taillefer, R´ emi Tachet des Combes, and Ioannis´ Mitliagkas. Adversarial score matching and improved sampling for image generation. September 2020.

Richard Jordan, David Kinderlehrer, and Felix Otto. The variational formulation of the fokker– planck equation. SIAMjournal on mathematical analysis, 29(1):1–17, 1998.

Tero Karras, Samuli Laine, and Timo Aila. A Style-Based generator architecture for generative adversarial networks. arXiv preprint arXiv:1812.04948, December 2018.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyz ing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8110–8119, 2020.

Diederik P Kingma and Max Welling. Auto-Encoding variational bayes. arXiv preprint arXiv:1312.6114v10, December 2013.

Daniel Levy, Matthew D Hoffman, and Jascha Sohl-Dickstein. Generalizing hamiltonian monte carlo with neural networks. arXiv preprint arXiv:1711.09268, 2017.

Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. arXiv preprint arXiv:1610.03483, October 2016.

Radford M Neal et al. Mcmc using hamiltonian dynamics. Handbook of markov chain monte carlo, 2(11):2, 2011.

Alejandro F Queiruga, N Benjamin Erichson, Dane Taylor, and Michael W Mahoney. Continuousin-depth neural networks. arXiv preprint arXiv:2008.02389, 2020.

Martin Raphan and Eero P Simoncelli. Least squares estimation without priors or supervision. Neural computation, 23(2):374–420, February 2011. ISSN 0899-7667, 1530-888X.

Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. arXiv preprint arXiv:1505.05770, May 2015.

Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082, 2014.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer assisted intervention, pp. 234–241. Springer, 2015.

Tim Salimans, Diederik P Kingma, and Max Welling. Markov chain monte carlo and variational inference: Bridging the gap. arXiv preprint arXiv:1410.6460, October 2014.

Ken Shoemake. Animating rotation with quaternion curves. In Proceedings of the 12th annual conference on Computer graphics and interactive techniques, pp. 245–254, 1985.

Jascha Sohl-Dickstein, Eric A Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. arXiv preprint arXiv:1503.03585, March 2015.

Jiaming Song, Shengjia Zhao, and Stefano Ermon. A-nice-mc: Adversarial training for mcmc. arXiv preprint arXiv:1706.07561, June 2017.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. arXiv preprint arXiv:1907.05600, July 2019.

Yang Song and Stefano Ermon. Improved techniques for training Score-Based generative models. arXiv preprint arXiv:2006.09011, June 2020.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. WaveNet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, September 2016a.

Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759, January 2016b.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.

Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, May 2016.

Shengjia Zhao, Hongyu Ren, Arianna Yuan, Jiaming Song, Noah Goodman, and Stefano Ermon. Bias and generalization in deep generative models: An empirical study. In Advances in Neural Information Processing Systems, pp. 10792–10801, 2018.

## A NON-MARKOVIAN FORWARD PROCESSES FOR A DISCRETE CASE

In this section, we describe a non-Markovian forward processes for discrete data and corresponding variational objectives. Since the focus of this paper is to accelerate reverse models corresponding to the Gaussian diffusion, we leave empirical evaluations as future work.

For a categorical observation x that is a one-hot vector with K possible values, we define the forward process as follows. First, we have q(x<sub>t</sub>|x<sub>0</sub>) as the following categorical distribution:

![](images/3b7faeb6feffd07b233c4c7e5c4fa7ac2356fde40b3e639f41a68070df46cbe5.jpg)

(17)

where 1<sub>K</sub> ∈ R<sup>K</sup> is a vector with all entries being 1/K, and α<sub>t</sub> decreasing from α<sub>0</sub> = 1 for t = 0 to α<sub>T</sub> = 0 for t = T. Then we define q(x<sub>t−1</sub>|x<sub>t</sub>, x<sub>0</sub>) as the following mixture distribution:

![](images/5708491d80b8c08294957919e9b46157f4068a6378e8be5e2a6ba67cb5287e02.jpg)

(18)

or equivalently:

![](images/5e0af4b3106405a81763da537f6ff7d558b81924c1854e7d1e261e6712e9727e.jpg)

(19)

which is consistent with how we have defined q(x<sub>t</sub>|x<sub>0</sub>).

Similarly, we can define our reverse process p<sub>θ</sub>(x<sub>t−1</sub>|x<sub>t</sub>) as:

![](images/18a4346b7b60da244cbd78cf4428b9a6f799f1ea00e63fe88f8f526cbae347dc.jpg)

(20)

where f<sup>(t)</sup>(x<sub>t</sub>) maps x<sub>t</sub> to a K-dimensional vector. As (1 − α<sub>t−1</sub>) − (1 − α<sub>t</sub>)σ<sub>t</sub> → 0, the sampling process will become less stochastic, in the sense that it will either choose x<sub>t</sub> or the predicted x<sub>0</sub> with high probability. The KL divergence

![](images/098c5e56817f63bb86c7894cbdde1642f31a6549c735732176efc27bc4b39f6d.jpg)

(21)

is well-defined, and is simply the KL divergence between two categoricals. Therefore, the resulting variational objective function should be easy to optimize as well. Moreover, as KL divergence is convex, we have this upper bound (which is tight when the right hand side goes to zero):

![](images/aeaa6a75477d99e8b48793993f516eb9ced288748370317d8ba13a56e096f9ed.jpg)

The right hand side is simply a multi-class classification loss (up to constants), so we can arrive at similar arguments regarding how changes in σ<sub>t</sub> do not affect the objective (up to re-weighting).

## B PROOFS

Lemma 1. For q<sub>σ</sub>(x<sub>1:T</sub>|x<sub>0</sub>) defined in Eq. (6) and q<sub>σ</sub>(x<sub>t−1</sub>|x<sub>t</sub>, x<sub>0</sub>) defined in Eq. (7), we have:

![](images/bea5f94af1da6794e5e4dbd6e0ac11c2835d071493cc336aa29eb6713fc91488.jpg)

(22)

<sub>Proof.</sub> <sub>Assume</sub> <sub>for</sub> <sub>any</sub> <sub>t</sub> <sub>≤</sub> <sub>T,</sub> <sub>qσ(xt|x0)</sub> <sub>=</sub> N <sub>(</sub>√<sub>αtx0, (1</sub> <sub>−</sub> <sub>αt)I)</sub> <sub>holds,</sub> <sub>if:</sub>

![](images/997bbbd9bc64bf1850712cb7feef5071618c5d6805b5509d27ee2d8f097a2e3b.jpg)

(23)

then we can prove the statement with an induction argument for t from T to 1, since the base case (t = T) already holds.

First, we have that

![](images/7d25e05bce1550e2790229c4f8f65bb3b9ebf61fcc334dd527a15557f5ea430c.jpg)

and

![](images/73497530f983285a8f5051f9b96b7931a13ef0f25a6a24616d50568321f02cb3.jpg)

(24)

![](images/b6f37b169fd8f1f46078db792f5f35ee8945ec599c854161736cf52ddd462dca.jpg)

(25)

From Bishop (2006) (2.115), we have that q<sub>σ</sub>(x<sub>t−1</sub>|x<sub>0</sub>) is Gaussian, denoted as N (µ<sub>t−1</sub>, Σ<sub>t−1</sub>) where

![](images/d55ad58f30a0cd3cf891ca01f6dfd2c9f95405d507bc2ae5b1009c86c8200f8b.jpg)

(26)

![](images/327bbc1c0f525f625c4aade83f8d82f0263090fe2ee5d4454782807ddf19df8a.jpg)

(27)

and

![](images/ffc79e143bafdd8fb92dfab183a7308a124b116fc3777892dba3e6a4c4fe5888.jpg)

(28)

<sub>Therefore,</sub> q<sub>σ</sub>(x<sub>t−1</sub>|x<sub>0</sub>) = N (√α<sub>t−1</sub>x<sub>0</sub>, (1 − α<sub>t−1</sub>)I)<sub>,</sub> <sub>which</sub> <sub>allows</sub> <sub>us</sub> <sub>to</sub> <sub>apply</sub> <sub>the</sub> <sub>induction</sub> argument. □

Theorem 1. For all σ > 0, there exists γ ∈ R<sup>T</sup> and C ∈ R, such that J<sub>σ</sub> = L<sub>γ</sub> + C.

Proof. From the definition of J<sub>σ</sub>:

![](images/df6c5e5dde3f936749758e7e9a1ee4cf735e7491b68cefc192888a302cfa77c6.jpg)

(29)

![](images/3c13f2bf52fde222bf7fe12b0e05536750fd14eca13cc33f8adf866a2df76cf1.jpg)

where we use ≡ to denote “equal up to a value that does not depend on <sub>θ</sub> (but may depend on q<sub>σ</sub>)”. For t > 1:

![](images/a389cc96d328f8c916c7e854c055ace65376a66e47bcb201b44408f59ec093a5.jpg)

(30)

![](images/d16e575aeb0f33fee1c202f404426d5ed0733dcb9938d703a3833b701e4270a1.jpg)

(31)

![](images/5bddd773f2aca53f40d8f7171f6060a675dbdfa9172bda5ec0b9136d3b12437d.jpg)

(32)

where d is the dimension of x<sub>0</sub>. For t = 1:

![](images/87af55e6f3feccdc6810daa5eee1304c3457b92cdd66030cc61a0fa2f72d9f4b.jpg)

(33)

![](images/0f8c3c2dde463671a98c616a83c99bfa44cda2ff27aa7c0a912ed9007347ef69.jpg)

(34)

Therefore, when γ<sub>t</sub> = 1/(2dσ<sup>2</sup>α<sub>t</sub>) for all t ∈ {1, . . . , T}, we have

![](images/1460d897c92f3db585fddc5a893248ee8b240836ebe620a28d33ca5be82a86e3.jpg)

(35)

for all <sub>θ</sub>. From the definition of “≡”, we have that J<sub>σ</sub> = L<sub>γ</sub> + C.

ODE corresponding to the “Variance-Exploding” SDE in Song et al. (2020).

Proof. In the context of the proof, we consider t as a continous, independent “time” variable and x and α as functions of t. First, let us consider a reparametrization between DDIM and the VE-SDE<sup>8</sup> by introducing the variables x¯ and σ:

![](images/0779e4df30336066655d4379e3088a75a157b463e1e9cd1070f5967b1621c06d.jpg)

(36)

for t ∈ [0, ∞) and an increasing continuous function σ : R<sub>≥0</sub> → R<sub>≥0</sub> where σ(0) = 0.

We can then define α(t) and x(t) corresponding to DDIM case as:

![](images/aaeda5d3b14365eb4f55c7f23dd182579269b0cdab837b2e50dfc429bfef95ff.jpg)

(37)

![](images/02e13a2d7c30b76cb43eb170b1042f9fd9c902ecebccbb849f73f03cabb73866.jpg)

(38)

This also means that:

![](images/a58848457c2c497b5723e8be4530bbe0aa5e44d0ce455009c64a93b40b5570c1.jpg)

(39)

![](images/be432194bfd9f9674839626ef467b1cebfc071f45da229034cc3813390cb51c2.jpg)

(40)

which establishes an bijection between (x, α) and (x¯, σ). From Equation (4) we have (note that α(0) = 1):

![](images/d7e089eac3b10f3c733d584d735b83fd9a8ef83fd9576d0de34ddf1df5f69c40.jpg)

(41)

which can be reparametrized into a form that is consistent with VE-SDE:

![](images/fc1a3e023358adc5a5382967d0dc106123cebe8dfc493156700720ce3c2e2aa9.jpg)

(42)

Now, we derive the ODE forms for both DDIM and VE-SDE and show that they are equivalent.

ODE form for DDIM We repeat Equation (13) here:

![](images/49ee1cea7a170a0b9f2f7050522c7db2b1267ee35ce3b71b8d3c299bc342c6e3.jpg)

(43)

which is equivalent to:

![](images/db858a08b990eb5d62c67f7c10f2fe2ddbbf93f0a4b7d5f67f56080b81b2274c.jpg)

(44)

Divide both sides by (−∆t) and as ∆t → 0, we have:

![](images/a7e5a128d05b15787432c8771a3eec86a503512b4c4729d63af598e4bb2346f0.jpg)

(45)

which is exactly what we have in Equation (14).

We note that for the optimal model,  (t) is a minimizer:

![](images/33a91525baeff42f11b60853841dc639423091687b486bc092383f45b23e7822.jpg)

(46)

where x(t) = pα(t)x(t) + p1 − α(t).

ODE form for VE-SDE Define p (x¯) as the data distribution perturbed with σ<sup>2</sup>(t) variance Gaussian noise. The probability flow for VE-SDE is defined as Song et al. (2020):

![](images/de7027d6b8449fa8399d09d03ac961ae92327c9c089843564ecb12f99da9e47b.jpg)

(47)

where g(t) = q <sup>dσ2(t)</sup> is the diffusion coefficient, and ∇<sub>x¯</sub> log p<sub>t</sub>(x¯) is the score of p<sub>t</sub>.

The σ(t)-perturbed score function ∇<sub>x¯</sub> log p<sub>t</sub>(x¯) is also a minimizer (from denoising score matching (Vincent, 2011)):

![](images/1ab5dde77f050ecc6d5141c55e16b0d001d47e871c770772f0e8c2ad90e7ca4e.jpg)

(48)

where x¯(t) = x¯(t) + σ(t).

Since there is an equivalence between x(t) and x¯(t), we have the following relationship:

![](images/8e1a39e46f257e8ac7994fd95c787b4e644214d952ed9151a0f7ffadd74a6d41.jpg)

(49)

from Equation (46) and Equation (48). Plug Equation (49) and definition of g(t) in Equation (47), we have:

![](images/5180a612cf6b2171ac183115c44327bbfa47454d01e947c698e07ea942833324.jpg)

(50)

and we have the following by rearranging terms:

![](images/13b13f1ec68a8f2235a74182457f00640a14dcb4fa734bc3a130700513f90103.jpg)

(51)

which is equivalent to Equation (45). In both cases the initial conditions are x¯(T) ∼ N (0, σ<sup>2</sup>(T)I), so the resulting ODEs are identical. □

## C ADDITIONAL DERIVATIONS

## C.1 ACCELERATED SAMPLING PROCESSES

In the accelerated case, we can consider the inference process to be factored as:

![](images/9cfd9e4c42739656751f2383e2257b305eae4eadcd5a741ccc1f4ffd1183c93f.jpg)

(52)

where τ is a sub-sequence of [1, . . . , T] of length S with τ<sub>S</sub> = T, and let τ¯ := {1, . . . , T} \ τ be its complement. Intuitively, the graphical model of {x<sub>τ</sub> }<sup>S</sup><sub>i=1</sub> and x<sub>0</sub> form a chain, whereas the graphical model of {x<sub>t</sub>}<sub>t∈τ¯</sub> and x<sub>0</sub> forms a star graph. We define:

![](images/5fc7ad05f88c3cfdd5789329b405d1b12029ef06ff2d3fc3ba5440befc0a5700.jpg)

(53)

![](images/c884600e9abc749f937434d59c1b8c6bcd7e86889f01fc48848f34c0c28cd022.jpg)

where the coefficients are chosen such that:

![](images/46f9686dd368e462564994f489153a2a2d5af9412ee895333332fd8071c2cf31.jpg)

(54)

i.e., the “marginals” match.

The corresponding “generative process” is defined as:

![](images/dcccace8f7b4db48be2499e91faff9a68e748455724c3dd5a736299e517983eb.jpg)

(55)

where only part of the models are actually being used to produce samples. The conditionals are:

![](images/8eef9b42a8c99072608ce74f7ff070e48c4eced95e89c9ae65ecbcebef1c698d.jpg)

(56)

![](images/43b332b56501620256511f1b5716118cc796d7a7841fb66bd16bb540c3100e94.jpg)

(57)

where we leverage q<sub>σ,τ</sub>(x<sub>τ</sub> |x<sub>τ</sub> , x<sub>0</sub>) as part of the inference process (similar to what we have done in Section 3). The resulting variational objective becomes (define x<sub>τ</sub> = ∅ for conciseness):

![](images/a4a23bcbc6682f507e800eb0db32b58eb050c1301fbb3dc8cc1586fa37fe7203.jpg)

(58)

![](images/bce25bcea3840b26b8c8a823ce9b89a3ebcf6e75131be706bede09d34491caa7.jpg)

(59)

![](images/ef485910fd9cd3f8666fe4fe548247da9eb8829fa82b9fd41934bc93dab7a61b.jpg)

where each KL divergence is between two Gaussians with variance independent of θ. A similar argument to the proof used in Theorem 1 can show that the variational objective J can also be converted to an objective of the form L .

## C.2 DERIVATION OF DENOISING OBJECTIVES FOR DDPMS

We note that in Ho et al. (2020), a diffusion hyperparameter β<sub>t</sub><sup>9</sup> is first introduced, and then relevant α to represent the variable α¯ in Ho et al. (2020) for three reasons. First, it makes it more clear that we only need to choose one set of hyperparameters, reducing possible cross-references of the derived variables. Second, it allows us to introduce the generalization as well as the acceleration case easier, because the inference process is no longer motivated by a diffusion. Third, there exists an isomorphism between α<sub>1:T</sub> and 1, . . . , T, which is not the case for β<sub>t</sub>.

In this section, we use β<sub>t</sub> and α<sub>t</sub> to be more consistent with the derivation in Ho et al. (2020), where

![](images/6068e688e4a870dcc5f8af1d3ee5a4efbc13151633c3d4dbe7850e230522d84e.jpg)

(60)

![](images/7189aae8a9d2c0d066e93d6e82eb5456f231ff17ec340aca5b41318708e75777.jpg)

(61)

can be uniquely determined from α<sub>t</sub> (i.e. α¯<sub>t</sub>).

First, from the diffusion forward process:

![](images/9ff70ef979d009df8c9f7f0380cc1b016651625e5f0b6956c6075cc878857518.jpg)

Ho et al. (2020) considered a specific type of p<sup>(t)</sup><sub>θ</sub> (x<sub>t−1</sub>|x<sub>t</sub>):

![](images/8ef9a9de389f2f6e92edd7d242a0b184ef0170f78f6397a0def1459088a773a1.jpg)

(62)

which leads to the following variational objective:

![](images/d350d5a813fb783e29ca6614a4d43af4f42ae1ff933962e7c80d53757813df44.jpg)

(63)

One can write:

![](images/a7ee32b301a14c3f7c64f43187bdf2fb1de26e6b8c54d237e3378985e63ed455.jpg)

(64)

Ho et al. (2020) chose the parametrization

![](images/1a905a273f720f5d7aef87d61fac215289d9d395c2392ef5a7ebe965d04904f6.jpg)

(65)

which can be simplified to:

![](images/a0326e129bb04d319275a25ebd6f8f574c13563d6c670049d1cb521dcee3113f.jpg)

(66)

## D EXPERIMENTAL DETAILS

## D.1 DATASETS AND ARCHITECTURES

We consider 4 image datasets with various resolutions: CIFAR10 (32 × 32, unconditional), CelebA (64 × 64), LSUN Bedroom (256 × 256) and LSUN Church (256 × 256). For all datasets, we set the hyperparameters α according to the heuristic in (Ho et al., 2020) to make the results directly comparable. We use the same model for each dataset, and only compare the performance of different generative processes. For CIFAR10, Bedroom and Church, we obtain the pretrained checkpoints from the original DDPM implementation; for CelebA, we trained our own model using the denoising objective L<sub>1</sub>.

Our architecture for  (t) (x<sub>t</sub>) follows that in Ho et al. (2020), which is a U-Net (Ronneberger et al., 2015) based on a Wide ResNet (Zagoruyko & Komodakis, 2016). We use the pretrained models from Ho et al. (2020) for CIFAR10, Bedroom and Church, and train our own model for the CelebA 64 × 64 model (since a pretrained model is not provided). Our CelebA model has five feature map resolutions from 64 × 64 to 4 × 4, and we use the original CelebA dataset (not CelebA-HQ) using the pre-processing technique from the StyleGAN (Karras et al., 2018) repository.

Table 3: LSUN Bedroom and Church image generation results, measured in FID. For 1000 steps DDPM, the FIDs are 6.36 for Bedroom and 7.89 for Church.  
![](images/38e1bd5bb9c40c42b9116f713415ab2cecdbacad16f65e301bff851035859035.jpg)

## D.2 REVERSE PROCESS SUB-SEQUENCE SELECTION

We consider two types of selection procedure for τ given the desired dim(τ ) < T:

• Linear: we select the timesteps such that τ<sub>i</sub> = bcic for some c;

• Quadratic: we select the timesteps such that τ<sub>i</sub> = bci<sup>2</sup>c for some c.

The constant value c is selected such that τ is close to T. We used quadratic for CIFAR10 and linear for the remaining datasets. These choices achieve slightly better FID than their alternatives in the respective datasets.

## D.3 CLOSED FORM EQUATIONS FOR EACH SAMPLING STEP

From the general sampling equation in Eq. (12), we have the following update equation:

![](images/04c49f98ed5840f241d8314ac8348ed9dddb2cc0a3f540b6a72a1a8f54d4efc9.jpg)

![](images/5c3ad215656d8a2cd4c65d84c0270d13362bce32cf7f8b165448981650c8acef.jpg)  
Figure 7: CIFAR10 samples from 1000 step DDPM, 1000 step DDIM and 100 step DDIM.

where

![](images/42e11f13424fae017a5f72a130a0d62f924fb5c5a83688017f425f7c4361bd75.jpg)

For the case of σˆ (DDPM with a larger variance), the update equation becomes:

![](images/77e23fa91c228bb680c510dcc1ec8b4231642004387500c879c86e93dd85aa84.jpg)

which uses a different coefficient for  compared with the update for η = 1, but uses the same coefficient for the non-stochastic parts. This update is more stochastic than the update for η = 1, which explains why it achieves worse performance when dim(τ) is small.

## D.4 SAMPLES AND CONSISTENCY

We show more samples in Figure 7 (CIFAR10), Figure 8 (CelebA), Figure 10 (Church) and consistency results of DDIM in Figure 9 (CelebA).

## D.5 INTERPOLATION

To generate interpolations on a line, we randomly sample two initial x<sub>T</sub> values from the standard Gaussian, interpolate them with spherical linear interpolation (Shoemake, 1985), and then use the DDIM to obtain x<sub>0</sub> samples.

![](images/7f248afd938e7bc7501145a9b2803ce494a16fe29ff8a3112ac068bd0d5d5af8.jpg)

(67)

(x<sup>(0)</sup><sub>T</sub> )<sup>></sup>x<sup>(1)</sup><sub>T</sub> (1) where θ = arccos kx<sup>(0)</sup><sub>T</sub> kkx<sup>(1)</sup><sub>T</sub> k These values are used to produce DDIM samples.

To generate interpolations on a grid, we sample four latent variables and separate them in to two pairs; then we use slerp with the pairs under the same α, and use slerp over the interpolated samples across the pairs (under an independently chosen interpolation coefficient). We show more grid interpolation results in Figure 11 (CelebA), Figure 12 (Bedroom), and Figure 13 (Church).

![](images/0c85d6aa14f320c91d24d280a4ef5df9329116b84898caa63cb843987046fd38.jpg)  
Figure 8: CelebA samples from 1000 step DDPM, 1000 step DDIM and 100 step DDIM.

![](images/3cae884f31ef872c596693673876acfa371b779727f9d79e9b70f6bad6b098d2.jpg)  
Figure 9: CelebA samples from DDIM with the same random x<sub>T</sub> and different number of steps.

![](images/76d741bcba543013d144567354e128325cb78583ddf01ca0681aeebb5fb116c6.jpg)  
Figure 10: Church samples from 100 step DDPM and 100 step DDIM.

![](images/f7b171e7b0eb967f90aa2c7697260d49279f7dd1c5f998dd50341e45764c537b.jpg)  
Figure 11: More interpolations from the CelebA DDIM with dim(τ ) = 50.

![](images/874826d612023bb14c63bbb8613c6269cb5f4522d8d31931b12b24ababeeb174.jpg)  
Figure 12: More interpolations from the Bedroom DDIM with dim(τ ) = 50.

![](images/54726b726c8a7fada3eb0253d086f8d2bf2e4cb6e0046d0cd83dc1072312503b.jpg)  
Figure 13: More interpolations from the Church DDIM with dim(τ) = 50.
# Constructive Counterfactuals

Inspired by Zheng Dai's work with _Ablation Based Counterfactuals_ (https://arxiv.org/abs/2406.07908), this repository presents some experiments.
First, we aim to reproduce Dai and Glifford's results. A slightly different setup was used, with a VAE rather than a diffusion model. Furthermore, rather than ablating by removing a model from an ensemble where each model was trained on a subset, we instead ablate simply by zeroing the parameters strongly activated by a specific sample. (see [ablate.py](ablate.py))

<img src="results/realistic_ablation.png" width="30%">

The first column represents ground truth. The second column represents the VAE's reconstruction before ablation. The third column represents the reconstruction after ablation. Note that ablation was only applied to the first sample. While the first row shows the decrease in quality, the second row further proves that other training samples are unaffected.

---

I have reproduced the results presented in _Ablation Based Counterfactuals_ with a VAE and experimented with a new idea, called _Constructive Counterfactuals_. This method draws off of ABCs, but presents the reverse: Instead of ablating a model to prevent it from learning from a specific piece of data, using gradient-based methods to manipulate the parameters to allow a model to generalize from new data in a single step, _without retraining_. Preliminary results show that it's possible, at least for VAEs, to quickly generalize to a new form of data with a single reverse ablation step.

I started by gathering a subset of MNIST containing only 512 samples. Within this subset, I hid all samples of class 0 when training. The VAE trained fine, but struggled to reconstruct images of class 0 after training, since they were excluded from the training data. Next, I applied

By applying just _a single gradient step on a single sample of class 0,_ we take the sign of the gradient and

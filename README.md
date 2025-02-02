# Constructive Counterfactuals

### By Jacob Buckhouse

### Part 1: Reproducing _Ablation Based Counterfactuals_

Inspired by Zheng Dai and David Glifford's work with _Ablation Based Counterfactuals_ ([arXiv:2406.07908v1](https://arxiv.org/abs/2406.07908)), this repository presents some experiments and expansions.
First, my aim was to reproduce their results. In order to encourage rapid experimentation and accessibility, I implemented the model as a VAE instead of a diffusion model. Additionally, rather than ablating by removing a model from an ensemble where each model was trained on a subset, I instead ablate simply by zeroing the parameters strongly activated by a specific sample: This allows these methods to be generalized to the most common architectures, rather than constructing an ensemble specifically for examining counterfactuals. (see [ablate.py](ablate.py) and [architecture.py](architecture.py))

<img src="results/realistic_ablation.png" width="30%">

The first column represents ground truth. The second column represents the VAE's reconstruction before ablation. The third column represents the reconstruction after ablation. Note that ablation was only applied to the first sample. While the first row shows the decrease in quality, the second row further proves that other training samples are unaffected.

Results:

- I was able to successfully reproduce Dai and Glifford's results in _Ablation Based Counterfactuals_, demonstrating that the process remains sound even with variation in architecture and exact implementation of the ablation.

---

### Part 2: Teaching an old model new tricks: _Constructive Counterfactuals_

Going further, I've begun experimenting with a new method, dubbed _Constructive Counterfactuals_. This method draws off of ABCs, but presents the reverse: Instead of ablating a model to prevent it from learning from a specific piece of data, we can add new data into a model _without retraining_ by using gradient-based methods to manipulate the parameters. Preliminary results show that it's possible, at least for VAEs, to quickly generalize to a new form of data with a single reverse ablation step.

---

I started by gathering a subset of MNIST containing only 512 samples. Within this subset, I hid all samples of class 0 when training. The VAE trained successfully, struggling to reconstruct images of class 0 after training since they were excluded from the training data, as expected. Next, took a single sample and essentially applied ablation in reverse: updating the parameters based off of the sign of the gradient multiplied by a constant. This method is similar to both FGSM and a step in the traditional gradient descent algorithm. However, unlike gradient descent, I only do this for a single step.

([constructive_counterfactuals.py](constructive_counterfactuals.py))

```python
def reverse_ablate(image, net,strength=0.001):
    net.eval()
    net.zero_grad()

    loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
    loss.backward()

    with torch.no_grad():
        for param in net.parameters():
            if param.grad is not None:
                param.data = param - torch.sign(param.grad) * strength
```

Notably, this step not only decreases loss and improves reconstruction on the one sample, but the model is able to generalize to other samples.

<img src="results/constructive_counterfactuals_example.png" width="30%">

The rows and columns follow the same conventions as the first figure. The third row now shows the increase in quality for samples of the same class that were _never shown to the model._ By reverse ablating just once with a single sample, the model was able to generalize and improve when given a similar task, while not degrading the rest of it.

Results:

- Initial results are promising that you can add data in a single gradient step rather than retraining. Models fine-tuned in this way seem to generalize reasonably well after exposure to even one new sample.

---

All code is available in this repository, and model weights are located at runs/vae_l5_linear_512_no0/ckpt/best.pt.

You should be able to reproduce my results by running `constructive_counterfactuals.py`

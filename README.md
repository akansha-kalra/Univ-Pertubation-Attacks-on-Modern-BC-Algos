# How Vulnerable is My Learned Policy?  Adversarial Attacks on Modern Behavior Cloning Algorithms #
<p align="center">
  <a href="https://arxiv.org/abs/2502.03698">View on ArXiv</a> |
  <a href="https://sites.google.com/view/uap-attacks-on-bc">Project Website</a>
</p>

<b> Akansha Kalra <sup>&ast;</sup>, Bavasagar Patil <sup>&ast;</sup> Guanhong Tao, Daniel S. Brown </b>

<sup>&ast;</sup> <i>Equal Contribution</i>

The project aims to develop Adversarial Attacks on Behavior Cloning Policies to check the adversarial robustness of these algorithms.
<b> <i>
- Vanilla Behavior Cloning
- LSTM-GMM
- Implicit Behavior Cloning (IBC)
- Diffusion Policy
- Vector Quantized-Behavior Transformer(VQ-BET)
</b></i>

# Environment setup
Begin by installing our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 22.04 ,
```console
$ conda env create -f uap_bc_env.yaml
```
And activate it by 
```console
$ conda activate adv_robodiff
```




The training dataset should go in `./data/robomimic/datasets/` and the checkpoints are stored in `/pre_trained_checkpoints/{task}_image/`, you can download pretrained policies as well as data from [here](https://diffusion-policy.cs.columbia.edu/data/). 

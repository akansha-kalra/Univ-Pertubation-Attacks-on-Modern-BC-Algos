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

The training dataset should go in `./data/robomimic/datasets/` and the checkpoints are stored in `/pre_trained_checkpoints/{task}_image/`, you can download pretrained DP, IBC and LSTM-GMM policies as well as data from [here](https://diffusion-policy.cs.columbia.edu/data/). 

We release our trained BC and VQBET policies [here](https://uofu.box.com/s/3zbwsrcc04l5exniew7r6tn64kwtqhz0). And our UAP attacks across all BC algorithms are present in the `final_UAP_patches` directory for all 3 tasks : Lift, Can, Square. For each task and BC algorithm, we trained and evaluated UAP patches using three different random seeds. The corresponding patches can be accessed at `final_UAP_patches/task/algo/train_{seed}/`. 


To train the UAP attack, for example on DP on Square task, use 
```console
python train.py  --config-name=Run1_BC_Untargeted_UAP_Square.yaml --config-dir=/diffusion_policy/config/
```


To evaluate the above white-box UAP attack against DP on the Square task, update patch_path to point to the desired UAP patch. You may use either a newly trained patch or one of our publicly released UAP patches at `final_UAP_patches/square/diffusion_polic_cnn/` 

```console
python Eval.py  --config-name=Eval_DP_1.yaml --config-dir=/diffusion_policy/Eval_config/
```


Similarly, to train and evaluate any attack across different behavior cloning (BC) algorithms, update the corresponding task configuration files located in `diffusion/policy/config/task/` to specify the appropriate attack settings and the trained BC checkpoint location.  
For reference, configuration files for training UAP attack on the Square task across all supported BC algorithms can be found `diffusion/policy/config/` .

### Note on Perturbation Budgets

Following the normalization scheme in the [original DDPM paper (Sec. 4)](https://arxiv.org/pdf/2006.11239), [Diffusion Policy (DP)](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py#L222) scales image observations from `[0, 1]` to `[-1, 1]` using the transformation `2x - 1`, whereas all other BC algorithms considered in our work operate in the original robomimic observation range of `[0, 1]`.

Thus, a perturbation budget of ε in DP's normalized observation space corresponds to ε/2 in the original observation space. To ensure equivalent perturbation budgets across all BC algorithms, we use **2ε for DP and ε for all other BC algorithms**.


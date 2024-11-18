# CF-AMC-SSL: Cost-Free Adversarial Multi-Crop Self-Supervised Learning
In this work, we explored the robustness of Extreme-Multi-Patch Self-Supervised Learning (EMP-SSL) against adversarial attacks using both standard and adversarial training techniques. Our findings underscored the significant
impact of multi-scale crops within the robust EMP-SSL algorithm, enhancing model robustness without sacrificing accuracy. This improvement contrasts with robust SimCLR, which relies on only a pair of crops per image and necessitates more training epochs. Moreover, we demonstrated the efficacy of incorporating free adversarial training into methods like SimCLR and EMP-SSL, even though training epochs are limited in EMP-SSL. This integration resulted in
the development of Cost-Free Adversarial Multi-Crop Self-Supervised Learning (CF-AMC-SSL), achieving substantial advancements in both robustness and accuracy while reducing training time. 
### SimCLR Overview:
- Uses data augmentations to create two views of each data point, treating these views as positive pairs during training.
- Involves a base encoder for feature extraction, a projection head for dimensionality reduction, and a contrastive loss to maximize similarity between positive pairs and minimize it for negatives.

### EMP-SSL Overview:
- Augments images into multiple sets of patches, which are passed through an encoder to extract features.
- The training objective combines Total Coding Rate (TCR) for regularization and invariance loss for consistency across augmentations.

### Adversarial Training in Both Frameworks:

- SimCLR: Generates one adversarial example per image and treats both augmented and adversarial versions as positive pairs.
- EMP-SSL: Generates multiple adversarial examples from different augmentations of the same image and aims to align these representations for better robustness.

### Training Efficiency:

 - EMP-SSL requires fewer epochs to converge and has a reduced runtime compared to SimCLR, despite using a higher number of patches in augmentation.


## An overview of the methodology
<p align="center">
<img src="./figures/Sim.jpg" alt="Alt Text" width="500">
 <br>
 <em><strong>The adversarially trained SimCLR vs. free adversarially trained
SimCLR framework.</strong></em>
</p>

<p align="center">
<img src="./figures/Emp.jpg" alt="Alt Text" width="500">
  <br>
  <em><strong>The adversarially trained crop-based EMP-SSL framework vs.
the free adversarially trained crop-based EMP-SSL (CF-AMC-
SSL).</strong></em>
</p>

<div>


## Running

### Standard Contrastive Learning
#### Pretraining Stage
(1) Standard Training
```
python main_simclr.py
 ```
 (2) Adversarial Training
```
python main_simclr_adv.py
 ```
(3) Free Adversarial Training
```
python main_simclr_free_adv.py
 ```
#### Evaluation Stage
```
python evaluate_simclr.py
 ```
### EMP-SSL: Extreme-Multi-Patch Self-Supervised-Learning  
#### Pretraining Stage
(1) Standard Training
```
python main_empssl.py
 ```
 (2) Adversarial Training
```
python main_empssl_adv.py
 ```
(3) Free Adversarial Training
```
python main_empssl_free_adv.py
```
#### Evaluation Stage
```
python evaluate_empssl.py
```
## Comparison of Models

**CF-AMC-SSL trains efficiently in fewer epochs, thereby reducing overall training time. By effectively employing multi-crop augmentations during base encoder training, it enhances both clean accuracy and robustness against PGD attacks.**  

### Evaluation Strategies

We evaluate two strategies:

- **Crop-Based Method**: This method involves random cropping of image augmentations, where the crop sizes range from 9×9 to 32×32 pixels.
- **Patch-Based Method**: This method uses fixed-scale patches as the primary method of augmentation.

<p align="center">
<img src="./figures/results.png" alt="Alt Text" width="800">
 <br>
 <em><strong>Note that the highest values are indicated in red, while the second highest values are highlighted in
blue.</strong></em>
</p>
<div>


### Comparison of Methods

We evaluate several methods on CIFAR-10, CIFAR-100, and training time. Note that the highest values are in **red**, and the second-highest values are in **blue**.

| **Models** | **CIFAR-10 Clean** | **CIFAR-10 PGD(4/255)** | **CIFAR-10 PGD(8/255)** | **CIFAR-100 Clean** | **CIFAR-100 PGD(4/255)** | **CIFAR-100 PGD(8/255)** | **Time (min)** |
|------------|--------------------|-------------------------|-------------------------|---------------------|--------------------------|--------------------------|----------------|
| **Patch-based EMP-SSL** (16 patches, 30 epochs) | 61 | 37.65 | 16.95 | 39.26 | 14.38 | 4.22 | 530 |
| **Crop-based EMP-SSL** (16 crops, 30 epochs) | <span style="color:red;">76.55</span> | <span style="color:red;">53.3</span> | <span style="color:red;">28.49</span> | <span style="color:red;">51.71</span> | <span style="color:red;">33.88</span> | <span style="color:red;">19.35</span> | 530 |
| **Crop-based SimCLR** (2 crops, 500 epochs) | 72.86 | 47.98 | 16.81 | 44.57 | 19.84 | 5.68 | 934 |
| **Patch-based SimCLR** (2 patches, 500 epochs)| 65.44 | 41.85 | 17.19 | 43.71 | 21.87 | 8.33 | 934 |
| **Patch-based EMP-FreeAdv** (16 patches, 10 epochs)| 61.83 | 42.28 | 21.53 | 40.31 | 23.78 | 12.13 | <span style="color:red;">97</span> |
| **Crop-based SimCLR-FreeAdv** (2 crops, 167 epochs)| 70.25 | 48.34 | 24.5 | 47.64 | 26.53 | 11.7 | <span style="color:blue;">157</span> |
| **Crop-based EMP-FreeAdv (CF-AMC-SSL)** (16 crops, 10 epochs)| <span style="color:blue;">75.88</span> | <span style="color:blue;">55.97</span> | <span style="color:blue;">33.34</span> | <span style="color:blue;">50.74</span> | <span style="color:blue;">31.73</span> | <span style="color:blue;">17.19</span> | <span style="color:red;">97</span> |



## Acknowledement
This repo is inspired by [CL-Robustness](https://github.com/softsys4ai/CL-Robustness/tree/main) and [EMP-SSL](https://github.com/tsb0601/EMP-SSL) repos.

# CF-AMC-SSL
In this work, we explored the robustness of Extreme-Multi-Patch Self-Supervised Learning (EMP-SSL) against adversarial attacks using both standard and adversarial training techniques. Our findings underscored the significant
impact of multi-scale crops within the robust EMP-SSL algorithm, enhancing model robustness without sacrificing accuracy. This improvement contrasts with robust SimCLR, which relies on only a pair of crops per image and necessitates more training epochs. Moreover, we demonstrated the efficacy of incorporating free adversarial training into methods like SimCLR and EMP-SSL, even though training epochs are limited in EMP-SSL. This integration resulted in
the development of Cost-Free Adversarial Multi-Crop Self-Supervised Learning (CF-AMC-SSL), achieving substantial advancements in both robustness and accuracy while reducing training time. 
## An overview of the methodology
<p align="center">
<img src="./figures/Sim.jpg" alt="Alt Text" width="500">
 <br>
 <em><strong>Figure 1(a) The adversarially trained SimCLR vs. free adversarially trained
SimCLR framework.</strong></em>
</p>

<p align="center">
<img src="./figures/Emp.jpg" alt="Alt Text" width="500">
  <br>
  <em><strong>Figure 1(b) the adversarially trained crop-based EMP-SSL framework vs.
the free adversarially trained crop-based EMP-SSL (CF-AMC-
SSL).</strong></em>
</p>

<div align="center">

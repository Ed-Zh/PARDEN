This repo contains the data and code (minimal demo) for our paper (accepted at ICML 2024): PARDEN, Can You Repeat That? Defending against Jailbreaks via Repetition
 

Paper: https://arxiv.org/abs/2405.07932

Blogpost:

PARDEN_data/ contains the benign and harmful datasets genearted by different models. The generation of harmful dataset is explained in detail in the paper, using both GCG[1] and prompt injection. 


PARDEN_notebook_minimal.ipynb demonstrates how to use PARDEN and tests its performance on the harmful strings generated by llama2-7b. 

[1] Zou et al. Universal and Transferable Adversarial Attacks on Aligned Language Models. https://arxiv.org/abs/2307.15043



# Miniformer
The source code of A **Mini**malist Trans**former** (Miniformer) for Brain Functional Networks Analysis

## Paper
**Miniformer: A Minimalist Transformer for Brain Functional Networks Analysis**

Jun Yang, Mengxue Pang, Mingxia Liu, and Lishan Qiao

## Datasets
We used the following datasets:

- ABIDE (Can be downloaded [here](http://fcon_1000.projects.nitrc.org/indi/abide/))
- REST-meta-MDD (Can be downloaded [here](http://rfmri.org/REST-meta-MDD))
- ADNI (Can be downloaded [here](https://adni.loni.usc.edu/))

## Dependencies
Miniformer needs the following dependencies:

- python 3.7.13
- torch == 1.8.0
- numpy == 1.21.5
- einops == 0.6.1
- scipy == 1.7.3
- sklearn == 1.0.2
- tqdm == 4.66.1
- d2l == 0.17.6

## Structure

    Miniformer_SM: Miniformer with temporal smoothness prior.
    Miniformer_SP: Miniformer with sparsity prior.

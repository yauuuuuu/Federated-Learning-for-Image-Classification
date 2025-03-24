# A Comparative Study of Federated Learning Algorithms in Dermatology: Assessing Feasibility for Medical Applications

## Overview
Federated learning (FL) has emerged as a promising paradigm for training machine learning models on decentralised medical datasets while preserving patient privacy. This study evaluates the efficacy of three FL algorithms—FedAvg, FedProx, and FedAdam—on the ISIC2019 dermatology dataset, a benchmark for skin lesion classification. Their performance are analysed under non-IID data distributions, class imbalance, and hyperparameter sensitivity, with a focus on practical feasibility for medical applications. The results demonstrate FedAvg’s robustness in heterogeneous environments, FedProx’s effectiveness in mitigating client drift, and FedAdam’s potential when tuned for medical data characteristics. The findings highlight FL’s viability for healthcare applications but underscore the need for algorithm adaptations to address domain-specific challenges like class imbalance and data heterogeneity.

## Dataset
The dataset used for this research is from [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019), which consists of dermoscopic images of skin lesions.

The images in this repository has been preprocessed, through re-sizing and re-colouring, using steps as proposed in [Flamby](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019).

## Code Structure
- 'Flower_FL_ISIC.ipynb': Python notebook for running FL simulations
- 'isic_dataset': Contains dermoscopic images along with metadata, including partitioning information and ground truth labels.

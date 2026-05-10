# PepHiFuse2: Threshold Predictor for Retention Time Based Quality Control

This repository extends the [PepHiFuse](https://github.com/lyotvincent/PepHiFuse) retention time (RT) prediction model with a **threshold prediction pipeline**. Instead of manually choosing an RT error cutoff for each dataset, a random forest regressor learns to predict the optimal threshold directly from the distribution of absolute differences between predicted and observed RTs. The module makes peptide‑spectrum match filtering hands‑off and objective across search engines and FDR estimation strategies.

## Pipeline Overview

**1. Defining RT error and optimal threshold**  
For a dataset containing N peptides, we compute the absolute error between PepHiFuse‑predicted RT and experimentally observed RT for every peptide. An optimal global threshold is defined as the cutoff that maximises the F1‑score when separating validated identifications from unvalidated ones. This threshold can only be determined if ground‑truth labels are available, which is why we learn to predict it from simpler data characteristics.

**2. Dataset‑level feature extraction**  
The distribution of absolute RT errors varies across experiments. To capture this variability, we summarise the error values using a fixed set of statistics: mean, median, standard deviation, 25th and 75th percentiles, minimum, and maximum. These few numbers form a feature vector that describes the shape of the whole error distribution without revealing any peptide‑specific information.

**3. Bootstrap resampling to create training data**  
Training a regressor requires many examples of feature vectors paired with their corresponding optimal thresholds. Since real datasets with validation labels are scarce, we apply bootstrap resampling on the available training data. For each resampled subset of peptides, we recompute the feature vector and recalculate the optimal threshold (by sweeping candidate cutoffs and picking the one with highest F1). Repeating this process yields a rich training set that mirrors the variation encountered in practice.

**4. Random forest regression**  
A random forest regressor is trained to predict the optimal threshold from the summary features. The model minimises mean squared error on the bootstrap‑generated pairs. The random forest is chosen for its ability to handle non‑linear relationships and its robust generalisation on modest‑sized feature–target collections.

**5. Inference and automatic filtering**  
When a new dataset arrives, we compute the same summary features from its absolute RT errors (which requires only running PepHiFuse to obtain predictions). The trained regressor outputs a single predicted threshold. Each peptide‑spectrum match is then classified as high‑confidence if its absolute RT error falls below the predicted cutoff; otherwise it is filtered out. No manual threshold tuning or external labels are needed at this stage.

## Performance

We evaluated the threshold predictor on search results from three search engines (Comet, MS‑GF+, X!Tandem) combined with three FDR control strategies (global, separate, two‑stage). In nearly all configurations the predicted threshold achieved accuracy and F1‑scores within 0.02 of the theoretical optimum. Paired bootstrap tests confirmed no statistically significant difference for most settings. Even when a tiny difference appeared, it had negligible practical impact. The pipeline maintains its reliability because the underlying PepHiFuse model provides high‑precision RT predictions, and the bootstrap‑based feature engineering captures the structure of RT error distributions.

## Summary

The threshold predictor turns PepHiFuse into a more automated quality control tool for neoantigen identification. By learning to infer the ideal RT error cutoff from a handful of dataset‑level statistics, it removes a subjective step from the proteogenomics workflow. The result is objective filtering that works across diverse FDR methods and search engines, directly supporting reliable neoantigen prioritisation for downstream immunotherapy research.

# Mars32K Clean–Dusty Subsets for Dusty Mars Rover Image Restoration

## Overview

This dataset provides the clean and dusty Mars rover image subsets used in our study on dusty Mars rover image restoration.

The images are derived from the Mars32K dataset. The original Mars32K dataset does not provide explicit clean/dusty labels. The clean and dusty categorization used here follows the preprocessing and classification procedure described by Ye et al. (2024).

In Ye et al. (2024), unsuitable images were first removed, and a pretrained CLIP model was then used to classify the remaining images. Image features were compared with the text prompts "clear" and "dust" using cosine similarity. Images showing higher similarity to the "clear" prompt were categorized as clean, whereas images showing higher similarity to the "dust" prompt were categorized as dusty.

The present release organizes the classified clean and dusty images used as the source data for the training and evaluation subsets in our experiments and provides a stable access point for reproducibility.

## Dataset Structure

The dataset contains 2,908 images in total:

- Training set:
  - 1,424 clean images
  - 1,424 dusty images

- Test set:
  - 30 clean images
  - 30 dusty images

The directory structure is:

```text
Mars_dataset/
├── train/
│   ├── clean/
│   └── dust/
└── test/
    ├── clean/
    └── dust/
```

## Data Source

The images originate from the Mars32K dataset, which contains Mars rover imagery acquired by the Curiosity rover Mastcam.

The original Mars32K hosting link used in previous publications is currently no longer accessible. This release therefore provides the classified subsets used in our study to improve reproducibility and long-term accessibility.

## Clean/Dusty Classification

The original Mars32K dataset is unlabeled with respect to atmospheric dust conditions.

The clean/dusty classification used in this dataset follows Ye et al. (2024):

1. Unsuitable images were removed during preprocessing.
2. A pretrained CLIP model was used to encode both images and textual descriptions.
3. The text prompts "clear" and "dust" were used as the two candidate categories.
4. Cosine similarity between each image representation and the two text representations was calculated.
5. Images with higher similarity to "clear" were classified as clean.
6. Images with higher similarity to "dust" were classified as dusty.

The present work uses these classified results as the basis for subsequent training and evaluation.

## Intended Use

This dataset is provided for research on:

- Mars rover image restoration
- Martian atmospheric dust degradation
- clean–dusty image translation
- image dehazing and restoration
- domain adaptation and generative modeling

## Citation

If you use this dataset, please cite the Zenodo dataset record and the following work describing the clean/dusty classification procedure:

Ye et al. (2024), IEEE Transactions on Geoscience and Remote Sensing.

DOI: 10.1109/TGRS.2024.3432601

The DOI of this dataset is:

10.6084/m9.figshare.33439624

## Notes

This dataset is a derived and organized subset based on Mars32K and the classification procedure described by Ye et al. (2024). It should not be interpreted as a newly acquired Mars rover image dataset or as providing manually verified atmospheric dust labels.

The clean/dusty categories in this release are model-assisted classifications derived from the CLIP-based procedure described above rather than manually annotated physical ground-truth labels.

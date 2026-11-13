# cs400_image_segmentation
CS399/400 senior comps project on image segmentation algorithms

## Folders:

### Segformer
This is our implementation of the Segformer semantic segmentation model in PyTorch. This is based on the architecture in [this](https://arxiv.org/abs/2105.15203) paper. More details in the `README.md` in `Segformer/`.

### U-Net
This is our implementation of the U-Net image segmentation model in PyTorch. It is based on the architecture in [this](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) paper.

### dust_turner
Small flask web app that uses dustification.py. It takes two random photos from the test sets of mars32k_drl and allows the user to adjust the sliders in dustification to an appropriate point. These results will be aggregated in a local sql database. This is a way for us to decide the ranges of acceptable values for randomly dustifying images.

### ai4mars-dataset-merged-0.6
This is the nasa ai4mars dataset:

https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars

### Mars_dataset
This is the mars32k dataset used in the DRL_UMDSR paper:

https://figshare.com/articles/dataset/Mars32K_Clean_Dusty_Subsets_for_Dusty_Mars_Rover_Image_Restoration/33439624?file=68224333 



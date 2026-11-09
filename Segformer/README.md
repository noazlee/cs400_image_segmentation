## Segformer
Handled by Noah Lee

Will be trying to implement and fine tune a pretrained segformer-b0 model:
https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512 

Segformer repo for reference:
https://github.com/NVlabs/SegFormer/

### Segformer Model

#### encoder.py

#### decoder.py

#### model.py

### Training/Testing

### train.py

### test.py

### visualization.py

### erf.py

Receptive field:
The receptive field is the set of input pixels that can affect it at all. For a plain CNN it grows with each conv and stride. For SegFormer it is already the whole image at stage 1, because every stage uses self-attention, which lets each position look at every other position.

Effective Receptive Field: Being able to see a pixel doesn't mean the network uses it. The ERF measures how much each input pixel actually affects that feature. The measure is the gradient: if nudging input pixel (x, y) changes the feature a lot, that pixel matters. In practice the ERF is usually concentrated near the center and fades out,

Get_input_grad: 
* Forward pass
* Take center pixel of the feature map 
* Calculate gradient of the input pixels - how much did input pixels affect this pixel in feature map
* Store and normalize



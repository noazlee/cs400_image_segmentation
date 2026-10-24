## Segformer
Handled by Noah Lee

Will be trying to implement and fine tune a pretrained segformer-b0 model:
https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512 

https://github.com/NVlabs/SegFormer/

Todo (Isaac):
- Add data augmentation - see how the paper did it 
    * Random rescaling, from 0.5x to 2.0x
    * Random horizontal flip - dont do vertical flips
    * Random crop to the training size (512)
    * Photometric distortion: brightness, contrast and saturation jitter -> dust?
- Add metrics - check how the paper deals with H/4 x W/4
- Training (Noah)
- Test.py

# cs400_image_segmentation
CS399/400 senior comps project on image segmentation algorithms

## Folders:
### DRL_UMDSR
This is the repository for the unsupervised learning model trained for cleaning/dustifying mars images.

https://github.com/phoenixtreesky7/DRL_UMDSR

Quick test:
Quick test:
python train.py --name quick --dataroot ../data/mars32k_drl/ \
   --which_model_netG dr_ca --dh_real --allmodel --ngf 32 --norm sswitch \
   --batchSize 2 --gpu_ids 0 --resize_or_crop crop --fineSize 256 \
   --lr_policy lambda --no_dropout \
   --max_dataset_size 40 --niter 2 --niter_decay 0 --print_freq 10 --display_freq 10

nvidia-smi -l 1

Full run:
python train.py --name mars32k_dust --dataroot ../data/mars32k_drl/ \
    --which_model_netG dr_ca --dh_real --allmodel --ngf 32 --norm sswitch \
    --batchSize 2 --gpu_ids 0 --resize_or_crop crop --fineSize 256 \
    --lr_policy lambda --no_dropout --save_epoch_freq 25 --display_freq 200 \
    2>&1 | tee train_mars32k_dust.log

Test model:
Testing a model
python test.py --name mars32k_dust --which_epoch latest \
--dataroot ../data/mars32k_drl/test/ \
--which_model_netG dr_ca --dh_real --allmodel --ngf 32 --norm sswitch --no_dropout \
--batchSize 1 --sb --gpu_ids 0 --how_many 30 \
--resize_or_crop crop --fineSize 480

### ai4mars-dataset-merged-0.6
This is the nasa ai4mars dataset:

https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars


### Mars_dataset
This is the mars32k dataset used in the DRL_UMDSR paper:

https://figshare.com/articles/dataset/Mars32K_Clean_Dusty_Subsets_for_Dusty_Mars_Rover_Image_Restoration/33439624?file=68224333 



## THIS IS NOT OUR FOLDER - ANY MODIFICATIONS TO THIS FOLDER MUST BE EXPLICITELY DOCUMENTED

### THIS REPOSITORY IS FROM THE PAPER "Unsupervised Martian Dust Storm Removal via Disentangled Representation Learning" BY Dong Zhao et al.
repo link: https://github.com/phoenixtreesky7/DRL_UMDSR/tree/main 

## Changes
1. Added `epdn/__init__.py` and `epdn/epdn_networks.py` (new files). `models/dr_uspv_model.py` imports `epdn.epdn_networks.VGGLoss`, but the upstream repo never shipped that package. `VGGLoss` and `Vgg19` are copied from https://github.com/phoenixtreesky7/EPDN `models/networks.py`
2. Removed `from invo.involution import Involution2d` from `models/dr_uspv_net.py`
3. Removed hardcoded `time.sleep` calls in `train.py` 
4. Removed hardcoded machine-specific settings like absolute Windows paths
5. Enabled logging output. Added `logging.basicConfig(level=logging.INFO)` at the start of `train.py` and `test.py` main
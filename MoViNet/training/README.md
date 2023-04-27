# Install training ENV

#### Install ENV Guide: 

``1:`` CUDA ENV:
```
# If upgrade cuda, need to install driver firstly, otherwise the installation of CUDA will fail.
sudo apt-get install -y nvidia-kernel-source-515    

wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run 
sudo sh cuda_11.7.0_515.43.04_linux.run
```

``2:`` training_extensions ENV: <br>
https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/quick_start_guide/installation.html

```
git clone https://github.com/openvinotoolkit/training_extensions.git
cd training_extensions
git checkout develop   # or your  branch
python -m venv .otx
source .otx/bin/activate
pip install --upgrade pip    # Avoid install openvino fail.
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e .[full] "Maybe fail, just run next step"

pip install tox
# -- need to replace '310' below if another python version needed
tox devenv venv/otx -e pre-merge-all-py3    # py310
source venv/otx/bin/activate

```
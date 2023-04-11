# Person vehicle bike detection model
OpenVINO fine-tune need cuda 11.1, but linux kernel should <=5.9, so recommend to install Ubuntu20.04.2.
But I can't setup it successfully. After install cuda, run nvidia-smi, and then tips: "no device found", and can't enter graphic interface.
So just install latest cuda 11.6 version.

After install cuda, need to set env to make sure nvcc work
```
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
```

## person-vehicle-bike-detection-2003

#### Setup training ENV based on Refer

Dependecies:
```
sudo apt install python3-pip
pip install --upgrade pip

Speed up install in China
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

```


```
cd person-vehicle-bike-detection\train
git clone https://github.com/openvinotoolkit/training_extensions.git
<!-- https://gitee.com/openvinotoolkit-prc/training_extensions.git -->
git checkout -b misc remotes/origin/misc
export OTE_DIR=`pwd`/training_extensions

git clone https://github.com/openvinotoolkit/open_model_zoo --branch develop
<!-- https://gitee.com/openvinotoolkit-prc/open_model_zoo.git --branch develop -->
export OMZ_DIR=`pwd`/open_model_zoo

cd training_extensions
pip install --upgrade pip   # Upgrade to latest.
pip3 install -e ote/
cd -

mkdir models_src
```

``Note`` GPU env setup fail[Only match torch1.8.1 version's cuda work], I hard code to CPU to verify pass. Please refer: https://download.pytorch.org/whl/torch_stable.html

#### Refer
[1] https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-vehicle-bike-detection-2003 <br>
[2] https://github.com/openvinotoolkit/training_extensions/blob/misc/models/object_detection/model_templates/person-vehicle-bike-detection/readme.md <br>
[3] https://github.com/openvinotoolkit/training_extensions/blob/misc/README.md

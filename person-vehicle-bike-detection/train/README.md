# Person vehicle bike detection model

## person-vehicle-bike-detection-2003

#### Setup training ENV based on Refer

```
cd person-vehicle-bike-detection\train
python3 -m venv python-env && source python-env/bin/activate
git clone https://github.com/openvinotoolkit/training_extensions.git
<!-- https://gitee.com/openvinotoolkit-prc/training_extensions.git -->
git checkout -b misc remotes/origin/misc
export OTE_DIR=`pwd`/training_extensions

git clone https://github.com/openvinotoolkit/open_model_zoo --branch develop
<!-- https://gitee.com/openvinotoolkit-prc/open_model_zoo.git -->
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
#!/usr/bin/env bash

echo "Start fine-tune model: vehicle-person-bike-detection-2002"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "SCRIPT_DIR=$SCRIPT_DIR"

# datumaro

export OTE_DIR=${SCRIPT_DIR}/training_extensions
export OMZ_DIR=${SCRIPT_DIR}/open_model_zoo

source $OTE_DIR/models/object_detection/venv/bin/activate

export MODEL_TEMPLATE=${SCRIPT_DIR}/models_src/template.yaml
export WORK_DIR=${SCRIPT_DIR}/work_dir
export SNAPSHOT=${WORK_DIR}/vehicle-person-bike-detection-2002-1.pth
# python ${SCRIPT_DIR}/training_extensions/tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}

# echo "====================================="
# echo "Convert pth model to IR"
# echo " -load-weights=${SNAPSHOT}"
# echo " -save-model-to=./model_ir/"

# # Change to work_dir
# cd ${WORK_DIR}
# python ./export.py --load-weights ${SNAPSHOT} --save-model-to ${SCRIPT_DIR}/model_ir/
# cd ${SCRIPT_DIR}    # Return

# My export script after init.
# python3 /home/xiping_dev/mygithub/all_models_demo_script/person-vehicle-bike-detection/train/training_extensions/external/mmdetection/tools/export.py model.py /home/xiping_dev/mygithub/all_models_demo_script/person-vehicle-bike-detection/train/work_dir/vehicle-person-bike-detection-2002-1.pth /home/xiping_dev/mygithub/all_models_demo_script/person-vehicle-bike-detection/train/model_ir/  --opset=11 openvino --input_format BGR

# If mo convert fail, here is my script
# python ../training_extensions/models/object_detection/venv/lib/python3.8/site-packages/openvino/tools/mo/mo.py  --input_model="/home/xiping_dev/mygithub/all_models_demo_script/person-vehicle-bike-detection/train/model_ir/model.onnx" --mean_values="[0, 0, 0]" --scale_values="[255, 255, 255]" --output_dir="/home/xiping_dev/mygithub/all_models_demo_script/person-vehicle-bike-detection/train/model_ir/" --output="boxes,labels" --input_shape="[1, 3, 512, 512]" --reverse_input_channels

# Test model
# python ../open_model_zoo/demos/object_detection_demo/python/object_detection_demo.py -m ../model_ir/model.xml -at ssd --adapter openvino -d CPU -t 0.5 -i /home/xiping_dev/barrier_test_image.jpg --raw_output_message

echo "====================================="
echo "Fine-tune"


#!/usr/bin/env bash

echo "Start fine-tune model: vehicle-person-bike-detection-2002"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "SCRIPT_DIR=$SCRIPT_DIR"

# Do 1_prepare_workpath.sh before training.
# Make sure exists: work_dir/train.py eval.py model.py ...

export OTE_DIR=${SCRIPT_DIR}/training_extensions
export OMZ_DIR=${SCRIPT_DIR}/open_model_zoo

source $OTE_DIR/models/object_detection/venv/bin/activate

export MODEL_TEMPLATE=${SCRIPT_DIR}/models_src/template.yaml
export WORK_DIR=${SCRIPT_DIR}/work_dir
export SNAPSHOT=${WORK_DIR}/vehicle-person-bike-detection-2002-1.pth

echo "====================================="
echo "Start fine-tune model vehicle-person-bike-detection-2002"
echo " -src model=${SNAPSHOT}"
echo " -save-model-to=./model_ir/"

# # Change to work_dir
# cd ${WORK_DIR}
# python ./export.py --load-weights ${SNAPSHOT} --save-model-to ${SCRIPT_DIR}/model_ir/
# cd ${SCRIPT_DIR}    # Return
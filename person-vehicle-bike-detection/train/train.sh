#!/usr/bin/env bash

echo "Start fine-tune model: vehicle-person-bike-detection-2003"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "SCRIPT_DIR=$SCRIPT_DIR"

# datumaro

export OTE_DIR=${SCRIPT_DIR}/training_extensions
export OMZ_DIR=${SCRIPT_DIR}/open_model_zoo

source $OTE_DIR/models/object_detection/venv/bin/activate

export MODEL_TEMPLATE=${SCRIPT_DIR}/models_src/template.yaml
export WORK_DIR=${SCRIPT_DIR}/work_dir
export SNAPSHOT=${WORK_DIR}/vehicle-person-bike-detection-2003.pth
python ${SCRIPT_DIR}/training_extensions/tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}

echo "====================================="
echo "Convert pth model to IR"
echo " -load-weights=${SNAPSHOT}"
echo " -save-model-to=./model_ir/"

# Change to work_dir
cd ${WORK_DIR}
python ./export.py --load-weights ${SNAPSHOT} --save-model-to ${SCRIPT_DIR}/model_ir/
cd ${SCRIPT_DIR}    # Return

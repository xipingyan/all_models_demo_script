source /opt/intel/oneapi/setvars.sh
source python-env/bin/activate
source ../../openvino/build/install/setupvars.sh

OV_GPU_DumpLayersRaw=1 OV_GPU_DumpLayersPath=./output_binary/ 
OV_GPU_Verbose=4 python3 yolov6_ov.py
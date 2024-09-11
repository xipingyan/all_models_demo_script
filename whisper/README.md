# README

#### Dependencies

    python3 -m venv python-env && source python-env/bin/activate
    pip install --upgrade pip
    pip install "transformers>=4.35" "torch>=2.1,<2.4.0" "torchvision<0.19.0" "onnx<1.16.2" "peft==0.6.2" --extra-index-url https://download.pytorch.org/whl/cpu
    pip install moviepy

#### Run

    Read a mp4 file. (downloaded_video.mp4)

    run.sh
    python3 whisper_openvino.py
    python3 whisper_pytorch.py
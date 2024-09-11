from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline

from utils import get_audio

import openvino as ov
core = ov.Core()

model_dir='whisper-tiny-stateful/'
device = 'CPU'

ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device=device)

processor = AutoProcessor.from_pretrained(model_dir)

pipe = pipeline(
    "automatic-speech-recognition",
    model=ov_model,
    chunk_length_s=30,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
)

output_file='downloaded_video.mp4'
inputs, duration = get_audio(output_file)

task='translate'
transcription = pipe(inputs, generate_kwargs={"task": task}, return_timestamps=True)["chunks"]
print("transcription=", transcription)
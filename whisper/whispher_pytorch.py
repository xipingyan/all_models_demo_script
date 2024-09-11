from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import pipeline
import torch
from utils import get_audio

PT_MODEL_ID='./whisper_pymodel/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(PT_MODEL_ID)

#  attn_implementation='flash_attention_2',
pt_model = AutoModelForSpeechSeq2Seq.from_pretrained(PT_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                                                     use_safetensors=True, cache_dir='data',).to(device)
pt_model.eval()

pipe = pipeline(
    "automatic-speech-recognition",
    model=pt_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
)

output_file='../downloaded_video.mp4'
inputs, duration = get_audio(output_file)

task='translate'
transcription = pipe(inputs, generate_kwargs={"task": task}, return_timestamps=True)["chunks"]
print("transcription=", transcription)
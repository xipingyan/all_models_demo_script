from moviepy.editor import VideoFileClip
from transformers.pipelines.audio_utils import ffmpeg_read
from utils import 

def get_audio(video_file):
    """
    Extract audio signal from a given video file, then convert it to float,
    then mono-channel format and resample it to the expected sample rate

    Parameters:
        video_file: path to input video file
    Returns:
      resampled_audio: mono-channel float audio signal with 16000 Hz sample rate
                       extracted from video
      duration: duration of video fragment in seconds
    """
    input_video = VideoFileClip(str(video_file))
    duration = input_video.duration
    audio_file = video_file + ".wav"
    input_video.audio.write_audiofile(audio_file, verbose=False, logger=None)
    with open(audio_file, "rb") as f:
        inputs = f.read()
    audio = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    return {
        "raw": audio,
        "sampling_rate": pipe.feature_extractor.sampling_rate,
    }, duration
# Import libraries
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import PIL

import tensorflow as tf
import tensorflow_hub as hub
import tqdm
import cv2
import os

from my_util import get_top_k, dec_video, get_top_k_streaming_labels, plot_streaming_top_preds_at_step
from my_util import plot_streaming_top_preds

mpl.rcParams.update({
    'font.size': 10,
})

# Get the kinetics 600 label list, and print the first few labels:
label_fn="kinetics_600_labels.txt"
if os.path.isfile(label_fn) is False:
  label_fn='https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
  labels_path = tf.keras.utils.get_file(
    fname='labels.txt',
    origin=label_fn
  )
labels_path=label_fn
print("label_fn=", label_fn)
labels_path = pathlib.Path(labels_path)

lines = labels_path.read_text().splitlines()
KINETICS_600_LABELS = np.array([line.strip() for line in lines])

print("KINETICS_600_LABELS=", KINETICS_600_LABELS[:20])
# %% Load gif animate
jumpingjack_path="jumpingjack.gif"
if os.path.isfile(jumpingjack_path) is False:
  jumpingjack_url = 'https://github.com/tensorflow/models/raw/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/jumpingjack.gif'
  jumpingjack_path = tf.keras.utils.get_file(
      fname='jumpingjack.gif',
      origin=jumpingjack_url,
      cache_dir='.', cache_subdir='.',
  )

# %%time
id = 'a2'
mode = 'stream'
version = '3'
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
# China mirror
hub_url=f'https://hub.tensorflow.google.cn/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'

print("model hub_url=", hub_url)
model = hub.load(hub_url)

# init_states = model.init_states(jumpingjack[tf.newaxis].shape)
cur_shape=(1, 13, 224, 224, 3)
init_states = model.init_states(cur_shape)

all_logits = []

# To run on a video, pass in one frame at a time
states = init_states
# for image in tqdm.tqdm(images):
idx=0

# Insert your video clip here
video_fn="0412_1.mov"
video_fn="0412_2.mov"
# video_fn="DSC_2342.AVI"
video_fn="927.MP4"
if os.path.isfile(video_fn) is False:
  print("NO exist file: ", video_fn)
  exit()

cap = dec_video(video_fn)

# for image in tqdm.tqdm(images):
while(True):
  frame, img=cap.get_one_frame()
  if frame is None:
    break
  
  frame=tf.reshape(frame, (1,1,224,224,3))
  print("--->process image:", idx)
  idx+=1
  # print("frame type=", type(frame), frame)
  # predictions for each frame
  logits, states = model({**states, 'image': frame})
  all_logits.append(logits)
  probs = tf.nn.softmax(logits, axis=-1)
  final_probs = probs[-1]
  print('Top_k predictions and their probablities\n')
  for label, p in get_top_k(final_probs, 2, KINETICS_600_LABELS):
    print(f'  {label:20s}: {p:.3f}')
  # cv2.imshow("img", img)
  # cv2.waitKey(0)

# concatinating all the logits
logits = tf.concat(all_logits, 0)
# estimating probabilities
probs = tf.nn.softmax(logits, axis=-1)

final_probs = probs[-1]
print('Final Top_k predictions and their probablities\n')
for label, p in get_top_k(final_probs, 5, KINETICS_600_LABELS):
  print(f'{label:20s}: {p:.3f}')

# For gif format, set codec='gif'
# media.show_video(plot_video, fps=3)
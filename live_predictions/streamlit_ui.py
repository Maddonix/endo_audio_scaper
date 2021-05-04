# pylint: disable=[import-error, no-member]
import streamlit as st
import json
import sounddevice as sd
import numpy as np
import queue
import pathlib
import os
import asyncio
from collections import namedtuple
import torch
from datetime import datetime as dt
import sys
import utils

cwd = pathlib.Path.cwd()
with open("config.json", "r") as file:
    config = json.load(file)

os.chdir(config["model_path"])
from src.models.audio_model_min import AudioLitModel
os.chdir(cwd)

cuda_device = 0
trained_model = AudioLitModel.load_from_checkpoint(checkpoint_path=config["ckpt_path"]).to(cuda_device)

device = config["device"]
classes = config["classes"]
channels = config["channels"]
samplerate = config["samplerate"]
duration = config["duration"]
batchsize_predict = config["batchsize_predict"]
target_frames = samplerate * duration
pred = [0 for _ in classes]
sample_difference = target_frames // batchsize_predict

audiostream_tuple = namedtuple("audiostream_tuple", ["data", "frames"])
arrays = [np.zeros((channels, target_frames)) for _ in range(batchsize_predict)]
datalist = {_: [] for _ in range(batchsize_predict)}
frames = [0 for _ in range(batchsize_predict)]
recording = [False for _ in range(batchsize_predict)]

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(audiostream_tuple(preprocess(indata.copy()), frames))
def preprocess(data):
    data = data.swapaxes(0,1)
    return data
def predict(array, model):
    array = array[np.newaxis, ...]
    batch = torch.Tensor(array).to(cuda_device)
    batch = batch / batch.max()
    pred = model(batch)
    pred = torch.sigmoid(pred).cpu().detach().numpy()
    # pred[pred>0.5] = True
    # pred[pred<=0.5] = False
    return pred

date_format_string = "%Y-%m-%d %H:%M:%S"
pred_logger = utils.prediction_logger(classes, date_format_string)

q = queue.Queue()

st.set_page_config(page_title="Live Classification", page_icon=None, initial_sidebar_state='auto')
st.header("Live Predictions")

placeholders_classes = {i: st.empty() for i,_ in enumerate(classes)}
for i in range(len(classes)):
    placeholders_classes[i].text(f"{classes[i]}")


async def listen_and_predict(placeholders_classes):
    warmup = True
    new_information = False
    with sd.InputStream(samplerate=samplerate, device=device,channels=channels, callback=callback):
        sum_frames = 0
        while warmup:
            audio_tuple = q.get()
            sum_frames += audio_tuple.frames
            n_recording = sum_frames // sample_difference
            recording[n_recording] = True
            
            for i in range(batchsize_predict):
                if recording[i] is True:
                    if (frames[i] + audio_tuple.frames) < target_frames:
                        frames[i] += audio_tuple.frames
                        datalist[i].append(audio_tuple.data)
                        
            if n_recording == (batchsize_predict - 1):
                warmup = False
        while True:    
            audio_tuple = q.get()
            for i in range(batchsize_predict):
                if (frames[i] + audio_tuple.frames) < target_frames:
                        frames[i] += audio_tuple.frames
                        datalist[i].append(audio_tuple.data)
                else:
                    new_information = True
                    tmp_array = np.concatenate(datalist[i], axis = 1)
                    dims = tmp_array.shape
                    arrays[i][...,:dims[1]] = tmp_array.copy()[...,:]
                    current_index = i
                    datalist[i] = [audio_tuple.data]
                    frames[i] = audio_tuple.frames
            if new_information is True:
                pred = predict(arrays[current_index].copy(), trained_model)
                pred_logger.log_values(pred[0])
                for i, _ in enumerate(pred[0]):
                    placeholders_classes[i].text(f"{classes[i]}: {round(_*100, 2)} %")
                new_information = False

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(listen_and_predict(placeholders_classes))
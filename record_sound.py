# -*- coding: UTF-8 -*-
import pyaudio
import wave
import time
import librosa  # pip install librosa
import numpy as np


def Record_sound(RECORD_SECONDS, wave_path=r'.\\output.wav', RATE=32000, CHANNELS=1, FORMAT=pyaudio.paInt16):
    # record_second录音的时间（秒）
    # wave_path 录音文件保存的地址
    # rate 采样频率
    # 22050 的采样频率是常用的, 44100已是CD音质, 超过48000或96000的采样对人耳已经没有意义。这和电影的每秒 24 帧图片的道理差不多
    # channels 音频采样通道数
    # format 数据流的格式
    CHUNK = 1024
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print('ready 3 second to record sound...')
    print('3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        # print('data is', data)
        frames.append(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wave_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def move_array(arr, dirction='left', step=1):
    new_arr = np.zeros(len(arr))
    if 'left' == dirction:
        new_arr[0:(len(new_arr)-step-1)]= arr[step:-1]
        new_arr[len(new_arr)-step-1] = arr[-1]
        new_arr[len(new_arr)-step:-1] = arr[0:step-1]
        new_arr[-1] = arr[step-1]
        # new_list = list[step:-1]+[list[-1]]+list[0:step]
    elif 'right' == dirction:
        new_arr[0:step-1] = arr[-step:-1]
        new_arr[step-1] = arr[-1]
        new_arr[step:-1] = arr[0:-step-1]
        new_arr[-1] = arr[-step-1]
    return new_arr


def change_sound(wavfile, new_path, new_dirction='left', move_step=10, save=1):
    # 增加样本量
    audio, fs = librosa.load(wavfile)
    audio[10:-10] = move_array(audio[10:-10], dirction=new_dirction, step=move_step)
    if 1 == save:
        librosa.output.write_wav(new_path, audio, fs)
    return audio


if __name__ == '__main__':
    # Record_sound(5)
    change_sound(r'output.wav',new_path=r".\\audio\\new_audio\\new_output.wav", new_dirction='left', move_step=100)
    # print('audio\n', audio)
    # print('fs\n', fs)

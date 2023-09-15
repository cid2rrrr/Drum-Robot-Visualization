import tkinter, subprocess, librosa, librosa.display, time, signal, threading, pyaudio, wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage

import onset_novelty

window = tkinter.Tk()
window.title("TEST")
window.geometry('1640x800+100+100')
window.resizable(False, False)

# plt.figure()
fig, ax = plt.subplots(figsize=(20, 4))
ax.set_position([0.01,0.05,0.98,0.90])
ax.axis('off')

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack()

button_value = True
state = False
sub = None
ext_flag = threading.Event()
bpm = 0
audio_frames = []

REC_FILE = './output.wav'

DURATION = 2
SR = 22050
H = 32
N = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 512 # 버퍼 크기 (프레임 수)
NFFT=1024
NOVERLAP=512




##### BPM #####

def detect_bpm():
    global bpm
    time.sleep(2)
    
    while True:
        if ext_flag.is_set():
            return
        
        y, _ = librosa.load(REC_FILE, sr=SR, mono=True)
        if len(y) == 0:
            continue
        y = librosa.util.normalize(y)
        # y = librosa.effects.trim(y)
        
        nov, _ = onset_novelty.compute_novelty_complex(y, Fs=SR, N=N, H=H, gamma=10)
        nov_smooth = ndimage.gaussian_filter1d(nov, sigma=2)
        
        bpmm = librosa.beat.tempo(onset_envelope=nov_smooth,
            sr=SR, hop_length=H, start_bpm=90, max_tempo=150, ac_size=4)[0]
        bpm = round(bpmm, 2)  # 소수점 둘째 자리까지 반올림
        time.sleep(1)

###############

def button_action():
    global state, sub, bpm
        
    if not state:
        global ext_flag
        ext_flag = threading.Event()
        t = threading.Thread(target=load_img)
        t.start()
        t2 = threading.Thread(target=detect_bpm)
        t2.start()
        sub = subprocess.Popen(["python", "record.py"])
        sub2 = subprocess.Popen(["python", "chop.py"])
        
        button['text'] = 'Stop'
        state = True
        
    elif state and sub is not None:
        ext_flag.set()
        sub.send_signal(signal.SIGTERM)
        
        button['text'] = 'Rec'
        state = False
        print(bpm)
    else:
        button['text'] = 'Reset'
        state = False
        sub = None


def load_img():
    global audio_frames, stream, canvas
    
    
    audio_frames = []
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SR, input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=3)

    while True:
        if ext_flag.is_set():
            return
        
        audio_data = stream.read(CHUNK)
        audio_frames.append(audio_data)
        
        imsi_arr = np.frombuffer(b''.join(audio_frames), dtype=np.int16).astype(np.float32)
        
        if len(imsi_arr) < SR * DURATION:
            # imsi_arr = np.append(np.zeros(SR*DURATION-len(imsi_arr)), imsi_arr, axis=0)
            imsi_arr = np.append(imsi_arr, np.zeros(SR*DURATION-len(imsi_arr)),  axis=0)
        elif len(imsi_arr) > SR * DURATION:
            imsi_arr = imsi_arr[-SR * DURATION:]
        
        mel = librosa.feature.melspectrogram(y=imsi_arr, sr=SR, n_mels=32, hop_length=128)
        mel = librosa.power_to_db(mel, ref=np.max)

        ax.clear()
        librosa.display.specshow(mel)
        canvas.draw()


button = tkinter.Button(window, overrelief='solid', width=15, command=button_action, text='Rec')
button.pack()


if __name__ == '__main__':

    window.mainloop()
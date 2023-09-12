import librosa, os, time
import numpy as np
import onset_novelty
from scipy import ndimage
from pathlib import Path

RATE = 22050
H = 32
N = 512

OUTPUT_FILENAME = 'output.wav'
BPM_PATH = './BPM/'

if __name__ == '__main__':
    try:
        while True:
            time.sleep(0.5)
            
            y, _ = librosa.load(OUTPUT_FILENAME, sr=RATE, mono=True)
            y = librosa.util.normalize(y)
            
            #### BPM을 파일명으로 전달
            
            nov, _ = onset_novelty.compute_novelty_complex(y, Fs=RATE, N=N, H=H, gamma=10)
            nov_smooth = ndimage.gaussian_filter1d(nov, sigma=2)
            
            bpm = librosa.beat.tempo(onset_envelope=np.append(np.array([0,0,0,0,0,0,0,0,0]),nov_smooth),
                sr=RATE, hop_length=H, start_bpm=90, max_tempo=150, ac_size=4)[0]
            bpm = round(bpm, 2)  # 소수점 둘째 자리까지 반올림
            
            if len(os.listdir(BPM_PATH)) > 0:
                os.remove(os.path.join(BPM_PATH, os.listdir(BPM_PATH)[0]))
            Path(os.path.join(BPM_PATH, str(bpm))).touch()
    except KeyboardInterrupt:
        pass
    
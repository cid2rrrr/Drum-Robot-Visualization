import librosa, soundfile, time, os
import numpy as np
from audiomentations import AddGaussianNoise
from matplotlib import pyplot as plt
import librosa.display
import onset_novelty
from scipy import ndimage


gauss = AddGaussianNoise(p=1.0, min_amplitude=0.0005, max_amplitude=0.001)

OUTPUT_FILENAME = 'output.wav'
BPM_PATH = './BPM/'

RATE = 22050
H = 32
N = 512



def imsi():
    num_used_peak = 0
    not_changed_cnt = 0
    y_len = 0
    bpm = 0
    chop_point = 0
    threshold = 0
    
    try:
        while True:
            time.sleep(0.5)
            
            start = time.time()
            
            y, _ = librosa.load(OUTPUT_FILENAME, sr=RATE, mono=True, offset=librosa.samples_to_time(chop_point*H))
            y = librosa.util.normalize(y)
            
            #### BPM을 파일명으로 전달
            
            nov, _ = onset_novelty.compute_novelty_complex(y, Fs=RATE, N=N, H=H, gamma=10)
            nov_smooth = ndimage.gaussian_filter1d(nov, sigma=2)
            
            # bpm = librosa.beat.tempo(onset_envelope=np.append(np.array([0,0,0,0,0,0,0,0,0]),nov_smooth),
            #     sr=RATE, hop_length=H, start_bpm=90, max_tempo=150, ac_size=4)[0]
            # bpm = round(bpm, 2)  # 소수점 둘째 자리까지 반올림
            
            # if len(os.listdir(BPM_PATH)) > 0:
            #     os.remove(os.path.join(BPM_PATH, os.listdir(BPM_PATH)[0]))
            # Path(os.path.join(BPM_PATH, str(bpm))).touch()
            
            #####
            nov, _ = onset_novelty.compute_novelty_complex(y, Fs=RATE, N=N, H=H, gamma=10)
            nov_smooth = ndimage.gaussian_filter1d(nov, sigma=8)
            peaks = onset_novelty.peak_picking_roeder(np.append(np.array([0,0,0,0,0]),nov_smooth), direction=None, abs_thresh=None, 
                                            rel_thresh=None, descent_thresh=None, 
                                            tmin=None, tmax=None)
            peaks = peaks[::-1] - 5
            
            onset_bt = librosa.onset.onset_backtrack(peaks, nov_smooth)
            #####
            
            onset_bt += chop_point
            onset_bt = onset_bt[onset_bt > threshold]
            # onset_bt = onset_bt[num_used_peak:]
            
            if len(onset_bt) >= 4: # at least 16 and plus 2
                onset_bt = onset_bt[:-2] - chop_point
                for i in range(len(onset_bt)):
                    j = 4134
                    if i < len(onset_bt)-1:
                        if (onset_bt*H)[i+1] - (onset_bt*H)[i] < 4134:
                            j = (onset_bt*H)[i+1] - (onset_bt*H)[i]
                    imsi = y[(onset_bt*H)[i]:(onset_bt*H)[i]+j]
                    if len(imsi) < 4134:
                        tmp = np.zeros(2000)
                        tmp = gauss(tmp, sample_rate=RATE)
                        imsi = np.append(imsi,tmp)
                        imsi = imsi[:4134]
                    imsi = librosa.util.normalize(imsi)
                    
                    f_sample = librosa.feature.melspectrogram(y=imsi, hop_length=128)
                    librosa.display.specshow(librosa.power_to_db(f_sample), cmap='jet', y_axis='mel')
                    plt.clim(-70,20)
                    plt.axis('off')
                    plt.savefig(fname='./imsi/mel/'+str(onset_bt[i]+chop_point).zfill(6)+'.jpg', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    soundfile.write('./imsi/'+str(onset_bt[i]+chop_point).zfill(6)+'.wav', imsi, RATE, format='WAV')
                
                num_used_peak += len(onset_bt)
                onset_bt += chop_point
                chop_point = int((onset_bt[-1] + onset_bt[-2])/2) +2
                threshold = onset_bt[-1]
                print(chop_point)
                print(threshold)
                    
            print(num_used_peak)
            
            if y_len == len(y):
                not_changed_cnt += 1
            
            if not_changed_cnt == 2:
                #####
                for i in range(len(onset_bt)):
                    j = 4134
                    if i < len(onset_bt)-1:
                        if (onset_bt*H)[i+1] - (onset_bt*H)[i] < 4134:
                            j = (onset_bt*H)[i+1] - (onset_bt*H)[i]
                    imsi = y[(onset_bt*H)[i]:(onset_bt*H)[i]+j]
                    if len(imsi) < 4134:
                        tmp = np.zeros(2000)
                        tmp = gauss(tmp, sample_rate=RATE)
                        imsi = np.append(imsi,tmp)
                        imsi = imsi[:4134]
                    imsi = librosa.util.normalize(imsi)
                    
                    f_sample = librosa.feature.melspectrogram(y=imsi, hop_length=128)
                    librosa.display.specshow(librosa.power_to_db(f_sample), cmap='jet', y_axis='mel')
                    plt.clim(-70,20)
                    plt.axis('off')
                    plt.savefig(fname='./imsi/mel/'+str(onset_bt[i]).zfill(6)+'.jpg', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    soundfile.write('./imsi/'+str(onset_bt[i]).zfill(6)+'.wav', imsi, RATE, format='WAV')
                
                num_used_peak += len(onset_bt)
                
                # chop_point = onset_bt[-1]
                # threshold = onset_bt[-1] +1
                print(chop_point)
                print(onset_bt)
                print(threshold)
                #####
            else:
                y_len = len(y)
                not_changed_cnt = 0
            
            if not_changed_cnt > 3:
                break
                        
            end = time.time()
            print(end - start)


    except KeyboardInterrupt:
        pass
            
            
def main():
    imsi()


if __name__ == "__main__":
    main()
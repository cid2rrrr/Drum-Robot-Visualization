import os, time, shutil, librosa
import model_def, onset_novelty
import numpy as np

MEL_PATH = './imsi/mel/'
DATA_PATH = './data/'
BPM_PATH = './BPM/'

IMG_HEI = 217
IMG_WID = 334

prd_cls = []

def run_cnn():
    
    arr = []
    prd_cls.clear()
    
    ####
    bpm = float(os.listdir(BPM_PATH)[0])
    ####
    peaks = [int(x[:-4]) for x in os.listdir(DATA_PATH)]
    time = librosa.samples_to_time(peaks * model_def.H)
    beat = onset_novelty.quantize_n_sec2beat(time, bpm)
    ####
    
    test_ds = model_def.tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH,
        shuffle=False,
        label_mode=None,
        image_size=(IMG_HEI, IMG_WID),
        batch_size=1)

    normalization_layer = model_def.tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    norm_test_ds = test_ds.map(lambda x: (normalization_layer(x)))
    
    prd = model_def.model.predict(norm_test_ds)
    
    for i in range(prd.shape[0]):
        if prd[i].max() > 0.7:
            arr.append(prd[i].argmax())
            # arr.append(prd[i].argmax()-1 if prd[i].argmax() >7 else prd[i].argmax())
        else:
            beat_pos = beat[i]
            top3_idx = prd[i].argsort()[-3:][::-1]
            beat_pos_start = max(0, beat_pos-31)
            
            starting_point = 0
            if beat_pos - 31 < 0:
                starting_point = abs(beat_pos - 31)
            
            bar = np.zeros((31,30))

            if beat_pos_start != beat_pos:
                for j in range(int(beat_pos_start), int(beat_pos)):
                    if j in beat:
                        bar[int(starting_point+j-beat_pos_start)][arr[list(beat).index(j)]] = 1
            
            for j in range(len(bar)):
                if np.where(bar[j] == 1)[0].size == 0:
                    bar[j][29] = 1

            # print(bar)
            rnn_prd = model_def.rnn_model.predict(np.expand_dims(bar, axis=0))[0][0]
            # print(rnn_prd)
            # max_val = -1
            # for candidate in range(len(rnn_prd)):
            #     if max_val < rnn_prd[candidate] * prd[i][candidate]:
            #         max_val = rnn_prd[candidate] * prd[i][candidate]
            #         chosen = candidate
                                
            # arr.append(chosen)######## imsi
            
            arr.append((prd[i] * rnn_prd).argmax())
    
    for a in arr:
        prd_cls.append(model_def.class_names[a])
    
    # transformed_prd_cls = []

    # for i, cls in enumerate(prd_cls):
    #     spl = cls.split('+')
    #     new_element = []

    #     for s in spl:
    #         if s == 'C':
    #             if i > 0 and any(keyword in prd_cls[i - 1] for keyword in ['R', 'MT', 'FT']) or any(keyword in prd_cls[i + 1] for keyword in ['R', 'MT', 'FT']):
    #                 new_element.append('CC_R')
    #             else:
    #                 new_element.append('CC_L')
    #         else:
    #             new_element.append(s)
        
    #     transformed_prd_cls.append('+'.join(new_element))
        
    # print(transformed_prd_cls)
    
    


if __name__ == "__main__":
    empty_cnt = 0
    prev_len = 0
    
    try:
        while True:
            
            time.sleep(0.5)
            
            files = os.listdir(MEL_PATH)
            files = [file for file in files if file.endswith('.jpg')]
            if len(files) == prev_len:
                empty_cnt += 1
            else:
                empty_cnt = 0
                prev_len = len(files)
            
            if empty_cnt > 2 and prev_len != 0:
                for file in files:
                    shutil.move(os.path.join(MEL_PATH, file), DATA_PATH)
                
                print(os.listdir(DATA_PATH))
                
                run_cnn()

                for i in range(len(files)):
                    print(librosa.samples_to_time(int(files[i][:-4])))
                    print(prd_cls[i])
                    print('---')
                print('----')
                
            elif empty_cnt > 10:
                break
            
            if len(files) > 12:
                for file in files:
                    shutil.move(os.path.join(MEL_PATH, file), DATA_PATH)
                
                print(os.listdir(DATA_PATH))
                
                run_cnn()

                for i in range(len(files)):
                    print(librosa.samples_to_time(int(files[i][:-4])))
                    print(prd_cls[i])
                    print('---')
                print('----')
                
            rm_files = os.listdir(DATA_PATH)
            for file in rm_files:
                os.remove(os.path.join(DATA_PATH, file))
            
                
    except KeyboardInterrupt:
        pass
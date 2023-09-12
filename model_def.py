from tensorflow.keras.applications import ResNet50 #, DenseNet121
# from tensorflow.keras.layers import Bidirectional, GRU, Dense
from audiomentations import AddGaussianNoise
import tensorflow as tf

N=512
H=32
SR=22050
img_height = 217
img_width = 334
gauss = AddGaussianNoise(p=1.0, min_amplitude=0.0005, max_amplitude=0.001)
class_names = ['B','B+C','B+CH','B+FT','B+HT','B+MT','B+OH','B+R','CH','CH+FT','CH+HT',\
    'CH+R','FT','HT','HT+FT','MT','OH','R','S','S+B','S+B+CH','S+B+FT','S+B+OH',\
    'S+B+R','S+C','S+CH', 'S+FT','S+OH','S+R']
pos = ['s','oh','ch', 'c', 'ht','mt','lt','r']



model = ResNet50(include_top=True, weights=None , input_shape=(img_height, img_width, 3), pooling=max, classes=29)
model.load_weights('./chkpnt/0106__/mel/cp-0008.ckpt')
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])
n_hidden=128
input_train=tf.keras.Input(shape=(31,30))
output_train=tf.keras.Input(shape=(1,29))

encoder_stack_h, encoder_last_h, encoder_last_c = tf.keras.layers.LSTM(
    n_hidden, activation='relu', 
    dropout=0.5, recurrent_dropout = 0.5,
    return_state = True, return_sequences=True)(input_train)

encoder_last_h = tf.keras.layers.BatchNormalization(momentum=0.6)(encoder_last_h)
encoder_last_c = tf.keras.layers.BatchNormalization(momentum=0.6)(encoder_last_c)

decoder_input = tf.keras.layers.RepeatVector(output_train.shape[1])(encoder_last_h)

decoder_stack_h = tf.keras.layers.LSTM(n_hidden, 
               activation = 'relu',
               dropout=0.5, 
               recurrent_dropout=0.5,
               return_sequences = True,
               return_state =False)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])

attention = tf.keras.layers.dot([decoder_stack_h, encoder_stack_h], axes = [2,2])
attention = tf.keras.layers.Activation('softmax')(attention)

context = tf.keras.layers.dot([attention, encoder_stack_h], axes = [2,1])
context = tf.keras.layers.BatchNormalization(momentum = 0.6)(context)

decoder_combined_context = tf.keras.layers.concatenate([context, decoder_stack_h])

out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_train.shape[2], activation='softmax'))(decoder_combined_context)

rnn_model = tf.keras.Model(inputs = input_train, outputs = out)

rnn_model.load_weights('C:/Users/KIST/Documents/GitHub/LSTM_Drummer/chkpnt/0216/single/cp-0061.ckpt')

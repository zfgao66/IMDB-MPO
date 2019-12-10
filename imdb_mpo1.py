import imdb_local
import numpy as np
import argparse
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,LSTM,Input,Reshape,Embedding
from keras.utils import to_categorical
from keras.preprocessing import sequence
from MPO_LSTM_all import MPO_LSTM_all

maxlen = 800
max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb_local.load_data(path='./data/imdb.npz',num_words=max_words)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# x_train = x_train.reshape(-1,1, maxlen).astype('float32')
print(x_train.shape)
# y_train = to_categorical(y_train,2)
r = 2

def MPO_lstm_model(pretrained_weights = None, input_size = (maxlen,)):
    inp = Input(input_size)
    out = Embedding(max_words, 128, input_length=maxlen, mask_zero=True)(inp)
    out = MPO_LSTM_all(w_input_shape=(2, 4, 4, 4), w_output_shape=(2, 4, 8, 4), w_ranks=(1, r, r, r, 1),
                         u_input_shape=(2, 4, 8, 4), u_output_shape=(2, 4, 8, 4), u_ranks=(1, r, r, r, 1),
                        activation='tanh', return_sequences=False, dropout=0.0, recurrent_dropout=0.0)(out)
    out = Dense(units=1, activation='sigmoid',kernel_initializer='glorot_normal')(out)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def train(args):
    EPOCH=200
    batch_size=300
    model = MPO_lstm_model()
    # model = TT_lstm_model()
    model_checkpoint = ModelCheckpoint(args['model'], monitor='val_loss', save_best_only=True)
    model.fit(x_train,y_train,batch_size=batch_size,callbacks=[model_checkpoint],epochs = EPOCH, verbose = 2)


    test_X, test_y = imdb_local.load_data(path='./data/imdb.npz',num_words=10000)[1]
    test_X = sequence.pad_sequences(test_X, maxlen=maxlen)
    # test_X = test_X.reshape(-1, 1, maxlen).astype('float32')
    # test_y = to_categorical(test_y, 2)
    loss, accuracy = model.evaluate(test_X, test_y, verbose=2)
    print('loss:%.4f accuracy:%.4f' % (loss, accuracy))


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default='123.hdf5',
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output loss plot")
    args = vars(ap.parse_args())
    return args

if __name__ == "__main__":
    args = args_parse()
    args['model'] = '123.hdf5'
    train(args)
import imdb_local
import numpy as np
import argparse
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,LSTM,Input,Reshape
from keras.utils import to_categorical
from keras.preprocessing import sequence

maxlen = 80
(x_train, y_train), (x_test, y_test) = imdb_local.load_data(path='./data/imdb.npz',num_words=10000)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_train = x_train.reshape(-1,1, 80).astype('float32')
print(x_train.shape)
y_train = to_categorical(y_train,2)


def lstm_model(pretrained_weights = None, input_size = (1,maxlen)):
    inp = Input(input_size)
    out = LSTM(units=256, activation='tanh',return_sequences=False)(inp)
    out = Dense(units=2, activation=None,kernel_initializer='glorot_normal')(out)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['acc'])
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def train(args):
    EPOCH=10
    batch_size=30
    model = lstm_model()
    # model = TT_lstm_model()
    model_checkpoint = ModelCheckpoint(args['model'], monitor='val_loss', save_best_only=True)
    model.fit(x_train,y_train,batch_size=batch_size,callbacks=[model_checkpoint],epochs = EPOCH, verbose = 2)


    test_X, test_y = imdb_local.load_data(path='./data/imdb.npz',num_words=10000)[1]
    test_X = sequence.pad_sequences(test_X, maxlen=maxlen)
    test_X = test_X.reshape(-1, 1, 80).astype('float32')
    test_y = to_categorical(test_y, 2)
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
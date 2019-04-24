import pickle
from math import ceil
from keras import Sequential
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Lambda,Bidirectional,Dropout
from keras import utils
import numpy as np
import json
from keras.models import load_model
def load_index(path=''):
    file = open(path+'word_set.pkl', 'rb')
    word_set = pickle.load(file)
    file.close()
    file = open(path+'word_index.pkl', 'rb')
    word_index = pickle.load(file)
    file.close()
    file = open(path+'index_word.pkl', 'rb')
    index_word = pickle.load(file)
    file.close()
    return word_set,word_index,index_word

def load_training_data(path=''):
    file = open(path+'input_data.pkl','rb')
    input_data = pickle.load(file)
    file.close()
    file = open(path+'output_data.pkl','rb')
    output_data = pickle.load(file)
    file.close()
    return input_data,output_data

word_set,word_index,index_word = load_index('no_ag_')
input_data, output_data = load_training_data('no_ag_')
model = Sequential()
label_size = len(word_set) + 1
feature_size = len(word_set) + 1

input_size = max([len(data) for data in input_data])
output_size = 1
#output_padding =[np.concatenate((np.zeros(input_size),np.pad(data, pad_width=output_size-len(data), mode='constant', constant_values=0)[output_size-len(data):])) for data in output_data]

print(input_size,output_size)


def data_generator(batch_size, input_data, output_data):

    while (True):
        for i in range(0, ceil(len(input_data) / batch_size)):
            index = i*batch_size
            bound = (i+1)*batch_size
            yield np.asarray(input_data[index:bound]), np.asarray(utils.to_categorical(output_data[index:bound], label_size) )
batch_size= 400

x,y= next(data_generator(batch_size,input_data,output_data))
print(x.shape)
print(y.shape)

model.add(Embedding(feature_size, 100,input_length=input_size))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(LSTM(256))
model.add(Dense(100, activation='relu'))
model.add(Dense(label_size , activation='softmax'))

model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])



model.fit_generator(data_generator(batch_size=batch_size,
                                   input_data=input_data,
                                   output_data=output_data),
                    samples_per_epoch=ceil(len(input_data)/batch_size),
                    nb_epoch=200,max_q_size=1000,verbose=1,nb_worker=1)


model.save('language_model_sentence_no_aggre.h5')

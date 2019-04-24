from keras.models import load_model
import pickle
import numpy as np
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Lambda,Bidirectional,Dropout

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

test = ['START_STATE', 'Poland', 'Harmonious', 'Red']
word_set,word_index,index_word = load_index('')

test_index = np.array([[word_index[x] for x in test]])

model = load_model('language_model_sentence.h5')
summary_leng = 30
for i in range (0,summary_leng):
    res=model.predict(test_index)
    next_word=index_word[np.argmax(res[0])]
    print(next_word)
    test.append(next_word)
    test_index = np.array([[word_index[x] for x in test[i+1:i+5]]])

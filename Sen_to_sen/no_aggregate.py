import pickle
import spacy
import json
def __aggregate_by_index(text,attribute):
    words=text.split(' ')
    result_text = ''
    for i,word in enumerate(words):
        if i == attribute['start']:
            result_text+=attribute['code']+' '
        elif i > attribute['start'] and i<=attribute['end']:
            continue
        else:
            result_text+=word+' '
    return result_text.strip()

def __aggregate_by_original(text,attribute):
    text = text.replace(attribute['original'], attribute['code'])
    return text

def __no_aggregate(text,attribute_list):

    corpus = []
    for i,attributes in enumerate(attribute_list):
        tmp_text = text[i]
        corpus.append(tmp_text)
    return corpus

def extract_attribute(tmp_dic):
    attribute_list = []
    summary_attribute = tmp_dic['attribute']
    text = tmp_dic['text']
    for sentence_attribute in summary_attribute:
        attribute_list.append(sentence_attribute)
    if len(attribute_list)==0:
        return None,None,None

    return __no_aggregate(text,attribute_list),attribute_list,text

def write_index(word_set,word_index,index_word,path=''):
    file = open(path+'word_set.pkl','wb')
    pickle.dump(word_set,file)
    file.close()
    file = open(path+'word_index.pkl','wb')
    pickle.dump(word_index,file)
    file.close()
    file = open(path+'index_word.pkl','wb')
    pickle.dump(index_word,file)
    file.close()


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

def word_to_index(corpus):
    word_set = set()
    word_index = dict()
    index_word = dict()

    for sentence in corpus:
        tokens = nlp(sentence)
        for token in tokens:
            if not token.is_punct:
                word_set.add(token.text)

    for i,word in enumerate(word_set):
        word_index[word] = i+1
        index_word[i+1] = word
    return word_set, word_index, index_word

def concat_corpus(text_set,attribute_set):
    corpus = []
    for i,text in enumerate(text_set):
        for j,sentence in enumerate(text):
            attribute_part = ''
            for attribute in attribute_set[i][j]:
                replace_attribute = attribute.replace(' ','_')
                attribute_part+=replace_attribute+' '
            result='START_STATE '+attribute_part+sentence+' END_STATE'
            corpus.append(result)
    return corpus

def convert_corpus(corpus):

    index_input = []
    for i,text in enumerate(corpus):
        tokens = nlp(text)
        index_text_list = []
        for token in tokens:
            if not token.is_punct:
                index_text_list.append(word_index[token.text])
        index_input.append(index_text_list)
    return index_input

def convert_training_data(index_corpus,window_size=4):
    input_data = []
    output_data = []

    for index_text in index_corpus:
        for i,index in enumerate(index_text):
            bound=i+window_size
            if bound >=len(index_text):
                continue
            input_data.append(index_text[i:bound])
            output_data.append(index_text[bound])
    return input_data,output_data

def write_training_data(input_data, output_data, path=''):
    file = open(path+'input_data.pkl','wb')
    pickle.dump(input_data, file)
    file.close()
    file = open(path+'output_data.pkl','wb')
    pickle.dump(output_data,file)
    file.close()

nlp = spacy.load('en_core_web_sm')
file = open('sml_summary.txt','r',encoding='utf-8')
all_data = []
for i,raw in enumerate(file):
    tmp_dic = json.loads(raw)
    output, input_attribute,_text = extract_attribute(tmp_dic)
    if output is None:
        continue
    data = dict()
    data['input'] = input_attribute
    data['output'] = output
    all_data.append(data)

Corpus = [data['output'] for data in all_data]
attributes_summary_list = []


for data in all_data:
    sentence_attribute_vale = []
    for sentence_attribute in data['input']:
        attributes_value = [single_attribute['value'] for single_attribute in sentence_attribute]
        sentence_attribute_vale.append(attributes_value)
    attributes_summary_list.append(sentence_attribute_vale)

corpus = concat_corpus(Corpus, attributes_summary_list)
for data in corpus:
    print(data)
word_set,word_index,index_word = word_to_index(corpus)
write_index(word_set,word_index,index_word,path='no_ag_')
#word_set,word_index,index_word = load_index('')

index_corpus = convert_corpus(corpus)
input_data,output_data = convert_training_data(index_corpus=index_corpus)

write_training_data(input_data,output_data,path='no_ag_')

def load_training_data(path=''):
    file = open(path+'input_data.pkl','rb')
    input_data = pickle.load(file)
    file.close()
    file = open(path+'output_data.pkl','rb')
    output_data = pickle.load(file)
    file.close()
    return input_data,output_data

import operator
import random
import data_loader as dl
import re
import numpy
import spacy
from copy import deepcopy

def tokenize_sentence(sentence):
    return [word.strip() for word in re.split('(\W+)?', sentence) if word.strip()]


def add_start_end(word_list):
    word_list.insert(0,'<STA>')
    word_list.append('<END>')
    return word_list

def get_stories(sentences_list):
    stories=[]
    for sentence in sentences_list:
        stories.append(add_start_end(tokenize_sentence(sentence['sample_sentence']))) 
        
    return stories

def get_pos_word_dict(word_pos_dict):
    pos_word_dict=dict()
    for key in word_pos_dict.keys():
        if word_pos_dict[key] in pos_word_dict.keys():
            pos_word_dict[word_pos_dict[key]].append(key)
        else:
            word_list=[]
            word_list.append(key)
            pos_word_dict[word_pos_dict[key]]=word_list
    return pos_word_dict

def get_pos_stories(sentences_list):
    nlp = spacy.load('en_core_web_sm')
    pos_stories=[]
    word_pos_dict=dict()
    for sentence in sentences_list:
        doc=nlp(sentence['sample_sentence'])
        pos_stories.append(add_start_end([token.pos_ for token in doc if token ]))

        for token in doc:
            word_pos_dict[token.text]=token.pos_
            
    return pos_stories,word_pos_dict



def set_next_pos_dict(pos_stories):
    next_pos_dict=dict()
    for pos_story in pos_stories:
        for i,pos in enumerate(pos_story):
            if i==len(pos_story)-1:
                break
            if pos not in next_pos_dict.keys():
                next_pos_list=[]
                next_pos_list.append(pos_story[i+1])
                next_pos_dict[pos]=next_pos_list
            else:
                if pos_story[i+1] not in next_pos_dict[pos]:
                    next_pos_dict[pos].append(pos_story[i+1])
    return next_pos_dict

def set_next_word_dict(stories):
    next_word_dict=dict()
    for story in stories:
        for i,word in enumerate(story):
            if i==len(story)-1:
                break
            if word not in next_word_dict.keys():
                next_word_list=[]
                next_word_list.append(story[i+1])
                next_word_dict[word]=next_word_list
            else:
                if story[i+1] not in next_word_dict[word]:
                    next_word_dict[word].append(story[i+1])
    return next_word_dict

def set_unigram_count_table(stories):
    count_table=dict()
    total_count=0
    for story in stories:
        for i,word in enumerate(story):
            if i==len(story)-1:
                break
            if word in count_table.keys():
                if story[i+1] in count_table[word].keys():
                    count_table[word][story[i+1]]=count_table[word][story[i+1]]+1
                elif story[i+1] not in count_table[word].keys():
                    count_table[word][story[i+1]]=1
            elif word not in count_table.keys():
                word_count_table=dict()
                word_count_table[story[i+1]]=1
                count_table[word]=word_count_table
            total_count+=1
    return total_count,count_table

def set_bigram_count_table(stories):
    count_table=dict()
    total_count=0
    for story in stories:
        for i,word in enumerate(story):
            if i==len(story)-1:
                break
            if i==0:
                continue
            if (story[i-1],word) in count_table.keys():
                if story[i+1] in count_table[(story[i-1],word)]:
                    count_table[(story[i-1],word)][story[i+1]]=count_table[(story[i-1],word)][story[i+1]]+1
                else: 
                    count_table[(story[i-1],word)][story[i+1]]=1
            elif (story[i-1],word) not in count_table.keys():
                word_count_table=dict()
                word_count_table[story[i+1]]=1
                count_table[(story[i-1],word)]=word_count_table
            total_count+=1

    return count_table

def set_trigram_count_table(stories):
    count_table=dict()
    total_count=0
    for story in stories:
        for i,word in enumerate(story):
            if i==len(story)-1:
                break
            if i<2:
                continue
            if (story[i-2],story[i-1],word) in count_table.keys():
                if story[i+1] in count_table[(story[i-2],story[i-1],word)]:
                    count_table[(story[i-2],story[i-1],word)][story[i+1]]=count_table[(story[i-2],story[i-1],word)][story[i+1]]+1
                else: 
                    count_table[(story[i-2],story[i-1],word)][story[i+1]]=1
            elif (story[i-2],story[i-1],word) not in count_table.keys():
                word_count_table=dict()
                word_count_table[story[i+1]]=1
                count_table[(story[i-2],story[i-1],word)]=word_count_table
            total_count+=1

    return count_table

def set_word_count(stories):
    word_count=dict()
    for story in stories:
        for word in story:
            if word in word_count.keys():
                word_count[word]+=1
            else:
                word_count[word]=1
    return word_count

def set_unigram_probability(unigram_count_table,single_word_count):
    unigram_probability_table=deepcopy(unigram_count_table)
    for key in unigram_probability_table.keys():
        for word_key in unigram_probability_table[key].keys():
            unigram_probability_table[key][word_key]/=single_word_count[key]
    return unigram_probability_table

def set_bigram_probability(bigram_count_table,unigram_count_table):
    bigram_probability_table=deepcopy(bigram_count_table)
    for key in bigram_probability_table.keys():
        for word_key in bigram_probability_table[key].keys():
            bigram_probability_table[key][word_key]/=unigram_count_table[key[0]][key[1]]
    return bigram_probability_table

def set_trigram_probability(trigram_count_table,bigram_count_table):
    trigram_probability_table=deepcopy(trigram_count_table)
    for key in trigram_probability_table.keys():
        for word_key in trigram_probability_table[key].keys():
            trigram_probability_table[key][word_key]/=bigram_count_table[(key[0],key[1])][key[2]]
    return trigram_probability_table

def reformat(sentence):
    sentence=sentence.replace(' \' ','\'')
    sentence=sentence.replace(' ?','?')
    sentence=sentence.replace(' !','!')
    sentence=sentence.replace(' %','%')
    sentence=sentence.replace(' ,',',')
    sentence=sentence.replace('# ','#')
    sentence=sentence.replace(' .','.')
    sentence=sentence.replace(' +','+')
    sentence=sentence.replace('< ','<')
    sentence=sentence.replace(' >','>')
    sentence=sentence.replace(' - ','-')
    sentence=sentence.replace('$ ','$')
    return sentence
	
DB=dl.DB_Connector()
sentences_list=DB.get_sentences()

vocab=set()
for sentence in sentences_list:
    vocab|=set(tokenize_sentence(sentence['sample_sentence']))

stories=get_stories(sentences_list)
pos_stories,word_pos_dict=get_pos_stories(sentences_list)
pos_word_dict=get_pos_word_dict(word_pos_dict)

pos_vocab=set()
for pos_sentence in pos_stories:
    pos_vocab|=set(pos_sentence)

word_idx=dict((word,i+1) for i,word in enumerate(vocab))
pos_idx= dict((pos,i+1) for i,pos in enumerate(pos_vocab))

single_word_count=set_word_count(stories)

total_count,unigram_count_table=set_unigram_count_table(stories)
bigram_count_table=set_bigram_count_table(stories)
trigram_count_table=set_trigram_count_table(stories)

pos_total_count,pos_count_table=set_count_table(pos_stories)

next_word_dict=set_next_word_dict(stories)
next_pos_dict=set_next_pos_dict(pos_stories)

unigram_probability_table = set_unigram_probability(unigram_count_table,single_word_count)
bigram_probability_table = set_bigram_probability(bigram_count_table,unigram_count_table)
trigram_probability_table = set_trigram_probability(trigram_count_table,bigram_count_table)

sentences=[x['sample_sentence'] for x in sentences_list if x['sample_sentence'].strip()]

has_created_sentences=dict()
jump_proability=0.3
#file=open('created_sentence.txt','w',encoding='utf-8')
nlp = spacy.load('en_core_web_lg')
while(len(has_created_sentences)<10000):
    probability=1.0
    previous_three_word='<STA>'
    previous_two_word=random.sample(next_word_dict['<STA>'],1)[0]
    probability*=unigram_probability_table[previous_three_word][previous_two_word]
    
    previous_one_word=random.sample(list(bigram_count_table[(previous_three_word,previous_two_word)]),1)[0]
    probability*=bigram_probability_table[(previous_three_word,previous_two_word)][previous_one_word]
    created_sentence=previous_two_word
    if previous_one_word=='<END>':
        created_sentence=reformat(created_sentence)
        if created_sentence not in sentences:
            has_created_sentences[created_sentence]=probability
        continue
    created_sentence+=' '+previous_one_word
    next_word=random.sample(list(trigram_count_table[(previous_three_word,previous_two_word,previous_one_word)]),1)[0]
    probability*=trigram_probability_table[(previous_three_word,previous_two_word,previous_one_word)][next_word]
    if next_word=='<END>':
        created_sentence=reformat(created_sentence)
        if created_sentence not in sentences:
            has_created_sentences[created_sentence]=probability
        continue
    created_sentence+=' '+next_word
    for j in range (15):
        previous_three_word=previous_two_word
        previous_two_word=previous_one_word
        previous_one_word=next_word
        if (numpy.random.uniform()>0.6) and ((previous_three_word,previous_two_word,previous_one_word) in trigram_count_table.keys()):
            next_word=max(trigram_count_table[(previous_three_word,previous_two_word,previous_one_word)].items(), key=operator.itemgetter(1))[0]
            probability*=trigram_probability_table[(previous_three_word,previous_two_word,previous_one_word)][next_word]
        else:
            next_word=random.sample(list(trigram_count_table[(previous_three_word,previous_two_word,previous_one_word)]),1)[0]
            probability*=trigram_probability_table[(previous_three_word,previous_two_word,previous_one_word)][next_word]
       
        if next_word=='<END>':
            break
        created_sentence+=' '+next_word
    if probability==1.0:
        continue
    created_sentence=reformat(created_sentence)
    if created_sentence not in sentences:
        has_created_sentences[created_sentence]=probability

		
file=open('created_sentence.txt','w',encoding='utf-8')
for key in has_created_sentences.keys():
    file.write(''.join(key)+'\t'+str(has_created_sentences[key])+'\n')
file.close()
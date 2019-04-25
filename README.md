[utility]
python sentence_generator.py

[dependency]
numpy - for random sampling when hit the random probability.
	
spacy - for get the pos information.

[environment]
The data loading should be editted with db information of data_loader.py before run the sentence_generator.py

User can replace the data_loader.py and change code
"""
DB=dl.DB_Connector()
sentences_list=DB.get_sentences()
"""
with your own training data.

If the application doesn't have enough training data, I recommend to use the probability version (Sentence_generator.py) instead of LSTM version.
The version of sentence_to_sentence(LSTM) sentence generator is under the folder Sen_to_sen.

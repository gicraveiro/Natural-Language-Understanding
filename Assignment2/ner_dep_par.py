'''
Assignment 2 - Natural Language Understanding Course - University of Trento

Author: Giovana Meloni Craveiro


Assigment is in the intersection of Named Entity Recognition and Dependency Parsing.

1. Evaluate spaCy NER on CoNLL 2003 data (provided)
    -> report token-level performance (per class and total)
        - accuracy of correctly recognizing all tokens that belong to named entities (i.e. tag-level accuracy)
    -> report CoNLL chunk-level performance (per class and total);
        - precision, recall, f-measure of correctly recognizing all the named entities in a chunk per class and total

2. Grouping of Entities. Write a function to group recognized named entities using noun_chunks method of spaCy. Analyze the groups in terms of most frequent combinations (i.e. NER types that go together).

3. One of the possible post-processing steps is to fix segmentation errors. Write a function that extends the entity span to cover the full noun-compounds. Make use of compound dependency relation.


'''
# IMPORTS

import nltk
import spacy
import pandas
#from spacy.tokens import Doc

# to import conll
import os
import sys
sys.path.insert(0, os.path.abspath('../data/'))

from conll import *

# FUNCTIONS

# Evaluate spaCy NER on CoNLL 2003 data (provided)
# report token-level performance (per class and total)
# report CoNLL chunk-level performance (per class and total);



# 2 - group recognized named entities using noun_chunks method of spaCy
# Analyze the groups in terms of most frequent combinations
def entity_grouping():
    return None
# 3 - extends the entity span to cover the full noun-compounds
def extend_entity_span(span):
    return None

# Useful functions

# extracts token texts and return them in a format of a list of lists, each list containing all tokens of the sentence
def group_noun_chunks(corpus):
    sentences = [[]]
    i = 0
    
    for tokenized in corpus:
        if ( tokenized != None):
            aux = tokenized.split()
            #print(aux)
            if (len(aux) >= 3 and aux[2].startswith("B-")):
                if (len(sentences) == 1 and len(sentences[0]) == 0):
                    i = 0
                else:
                    i = i + 1
                    sentences.append([])
                    #print(sentences[i-1])
                    #print("NEWLIST")
                    #print(aux[0],aux[2])
            sentences[i].append(aux[0])

    #for sent in sentences:   
    #    print (sent)
    #print(sentences)
    return sentences

def tokenized_back_to_string(data):

    list_of_lists_of_tokens = group_noun_chunks(data)  # one list containing many lists, that contain many strings, each one representing a token text
    list_of_sentences = [] # one list containing many strings, each one representing a sentence
    #print(list_of_lists_of_tokens)

    #Group tokens belonging to the same sentence into one string in the list of sentences
    for list in list_of_lists_of_tokens:
        str_format_sentence = ' '.join(list)
        list_of_sentences.append(str_format_sentence) 
    #print(list_of_sentences)

    # Groups the strings of all sentences into one single corpus string
    original_corpus_rebuilt = ""
    for str_format_sentence in list_of_sentences:
        original_corpus_rebuilt += str_format_sentence
    original_corpus_rebuilt = ' '.join(list_of_sentences)
    #print(original_corpus_rebuilt)

    return original_corpus_rebuilt

# Reconstruct tokens
#   merge tokens that were separated by hifens 
def reconstruct_hifenated_words(corpus):
    i = 0
    #print([(t.text, t.ent_iob_, t.ent_type_, t.whitespace_) for t in corpus])
    while i < len(corpus):
        if(corpus[i].text == "-"):
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+2])
            #print("RETOKENIZE",corpus[i-1:i+2],corpus[i-1],corpus[i],corpus[i+1])
            
            #i -= 1 # loop infinito
        
        else: 
            i += 1
        #print(corpus[i].text)


    #print(corpus)
    #print([(t.text, t.ent_iob_, t.ent_type_, t.whitespace_) for t in corpus])
    #print(type(corpus))
    return corpus
    
# MAIN

nlp = spacy.load("en_core_web_sm")
data = get_chunks("data/test.txt")

corpus = tokenized_back_to_string(data)
doc = nlp(corpus) # tokenize original reconstructed corpus but using spacy
# 1st step - reconstruction of the tokenization
doc = reconstruct_hifenated_words(doc) # addressing the hifen conversion issue
# 2nd step - adaptation to the proper format for evaluation - tip: use whitespace_ information

for token in doc:
    print([(token.text, token.ent_iob_, token.ent_type_, token.whitespace_)])

#evaluated_data = evaluate(data)
#print(evaluated_data)

# getting references (note that it is testb this time)
#refs = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
#print(refs[0])

#import nltk.tag.hmm as hmm
#trn_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
#hmm_model = hmm.HiddenMarkovModelTrainer()
#hmm_ner = hmm_model.train(trn_sents)
# getting hypotheses
#hyps = [hmm_ner.tag(s) for s in conll2002.sents('esp.train')]
#print(hyps[0])

#print(conll2002)

#results = evaluate(refs, hyps)

#pd_tbl = pandas.DataFrame().from_dict(results, orient='index')
#pd_tbl.round(decimals=3)

#hyps2 = [hmm_ner.tag(s) for s in data]
#print(hyps2[0])


'''
Passo a passo

1 - Leio os dados conll 2003 em formato conll com a função do conll.py CHECK
2 - Converte pro formato que processa direito com spacy
    - agregar cada uma das palavras/token em uma lista de sentenças? ou em uma grande lista de tokens? CHECK
3 - Processar com spacy CHECK
4 - Reconstruir o processo de tokenization, unindo os tokens com hífen CHECK
5 - Adaptar os tags - usar whitespace_


''' 


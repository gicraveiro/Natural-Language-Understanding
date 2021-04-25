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

# Evaluate spaCy NER on CoNLL 2003 data (provided)
# report token-level performance (per class and total)
# report CoNLL chunk-level performance (per class and total);
def evaluateA():
    return None
# group recognized named entities using noun_chunks method of spaCy
# Analyze the groups in terms of most frequent combinations
def entity_grouping():
    return None
# extends the entity span to cover the full noun-compounds
def extend_entity_span(span):
    return None

import nltk
import spacy
import pandas
from spacy.tokens import Doc
from spacy.util import compile_infix_regex
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS

#nlp = spacy.blank("en")
nlp = spacy.load("en_core_web_sm")
infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

# to import conll
import os
import sys
sys.path.insert(0, os.path.abspath('../data/'))

from conll import *

#nltk.download('conll2002')
#from nltk.corpus import conll2002
#c2002 = spacy.load('conll2002')

#print(len(conll2002.tagged_sents()))
#print(conll2002._chunk_types)
#print(conll2002.sents('esp.train')[0])
#print(conll2002.tagged_sents('esp.train')[0])
#print(conll2002.chunked_sents('esp.train')[0])
#print(conll2002.iob_sents('esp.train')[0])

#f = open("data/train.txt", "r")
#contents = f.read()
#print(contents)
#f.close()

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
    
def reconstruct(corpus):
    i = 0
    new_corpus = []
    flag = False
    for i in range (len(corpus)):
        if(corpus[i].text == "-"):
            #new_corpus[i-1] = 
            token.text = corpus[i-1].text + corpus[i].text + corpus[i+1].text
            
            flag = True
            print(new_corpus[i-1].text)
        else:
            if(flag == False): 
                new_corpus.append(corpus[i])
            flag = False
    #for token in corpus:
    #    if (token.txt == "-"):

    #return reconstructed
    


data = read_corpus_conll("data/test.txt")

data2 = get_chunks("data/test.txt")

print(data2)
'''
for sent in data2:
    if (sent != None):
        if(sent.split()[0] == "\n"):
            print("\n\n\n\n")
        else:
            print(sent.split()[0])
            print(sent)
            #break
'''
nk_list = group_noun_chunks(data2) # HAS PROBLEMS
newlist = []
#print(nk_list)
for list in nk_list:
    str_format = ' '.join(list)
    newlist.append(str_format) # LIST OF ALL SENTENCES IN STRING FORMAT
#print(newlist)

#final = ""
#for str_format in newlist:
#    final += str_format
#final = ''.join(newlist)
#print(final)


txt = "A sentence that actually makes sense in AL-AIN , United Arab Emirates 1996-12-06"


#for list in nk_list:
#    list = Doc(nlp.vocab, words=list)
#    print([(t.text, t.ent_iob_, t.ent_type_, t.whitespace_) for t in list])
#doc = Doc(nlp.vocab, words=nk_list)
#print([(t.text, t.ent_iob_, t.ent_type_, t.whitespace_) for t in (list in nk_list)])

test_txt = nlp(txt)
#nlp_proc = nlp(final)

#for t in nlp_proc:
#    print([t.text, t.ent_iob_, t.ent_type_, t.whitespace_])

print([(t.text, t.ent_iob_, t.ent_type_, t.whitespace_) for t in test_txt])

#reconstruct(test_txt)
'''
for sent in data:
    if (sent != None):
        if(sent[0] == "\n"):
            print("\n\n\n\n")
        else:
            print(sent[0])
            print(sent)

'''
# ALIGN OUTPUT TOKENIZATION TO CORRECT INPUT FORMAT OF THE EVALUATE FUNCTION
#   RECONSTRUCT TOKENIZATION AND ADAPT TAGS - tip: use whitespace_ information

# 1st step - reconstruction of the tokenization

# 2nd step - adaptation to the proper format for evaluation







#print(data)

#evaluated_data = evaluate(data)

#print(evaluated_data)

#print(c2002)

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
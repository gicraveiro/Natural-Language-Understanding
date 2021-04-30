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
import itertools
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
def entity_grouping(corpus):
    print("Entity grouping")
    
    list_of_groups = []
    ents_used = []
    counter = 0
    for chunk in corpus.noun_chunks:
        print(chunk.text)
        chunk_ents = []
        for token in chunk:        
            if (token.ent_type_ != ""):
                chunk_ents.append(token.ent_type_)
                ents_used.append(token.text)
                
        
        print(chunk_ents)
        list_of_groups.append(chunk_ents)

    #flag = "new"
    #for ent in corpus.ents:
    #    for used in ents_used:
    #        if (ent.text == used):
    #            flag = "match"
    #    if (flag != "match"):
    #        list_of_groups.append([ent])
    #        counter += 1
    #        print("new ent ",ent)   
    #    flag = "new"     
    #print(len(ents_used))
    #print(len(corpus.ents))
    #print(counter)
    #print(len(ents_used)+counter)
    #print(len(corpus.ents)-(len(ents_used)+counter))
    #c_ents = iter(corpus.ents)
    #for c_ent,used_ent in zip(c_ents,ents_used):
    #    for token in c_ent:
    #        if(token.text != used_ent):
    #            print(token.text, "!=", used_ent)
    #            list_of_groups.append([token.ent_type_])
    #            next(c_ents)
    #        else:
    #            print(token.text, "==", used_ent)


    classes = [] # name, frequency
    index = -1
    for list in list_of_groups:
        index = -1
        for item in classes:
            if(list == item[0]):
                index = 0
            #    classes[classes.index(list)][1] += 1
        if(index == -1):
            classes.append([list,0])         

    print("Frequency analysis of entity groups implementation in progress")
        #print("entity "+c_ent.text+" "+used_ent)
    #print(list_of_groups)
    return list_of_groups
# 3 - extends the entity span to cover the full noun-compounds
def extend_entity_span(corpus):
    print("Extended entity span is not implemented yet")
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
#   merge tokens that were separated by hyphens 
def reconstruct_hyphenated_words(corpus):
    i = 0
    #print([(t.text, t.ent_iob_, t.ent_type_, t.whitespace_) for t in corpus])
    while i < len(corpus):
        if(corpus[i].text == "-" and corpus[i].whitespace_ == ""):
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
    
# Printing tokens from both tokenizations for comparison reasons
def print_tokens_from_both_corpus_simultaneously(conll_data,spacy_doc):
    
    for conll_token,token in zip(conll_data,spacy_doc):
        properties = conll_token.split()
        print(properties[0] + ' -> ' + token.text)
        print(properties[-1] + ' -> ' + token.ent_iob_ +'-'+token.ent_type_+'\n')    

def print_processed_tokens_from_both_corpus_simultaneously(hyps,refs):
    
    for ref,hyp in zip(refs,hyps):
        print(ref,hyp)

def get_hyps(doc):
    hyps = []
    for token in spacy_doc:
        ent_type = token.ent_type_
        if(ent_type == 'PERSON'):
            ent_type = 'PER'
        elif(ent_type == 'O' or ent_type == 'LOC' or ent_type == 'ORG'):
            pass # doesnt change it
        else:
            ent_type = 'MISC'
    
        if(token.ent_iob_ == "O"):
            iob = "O"
        else: 
            iob = (token.ent_iob_ +'-'+ent_type)
        hyps.append((token.text,iob))
    #print(hyps)
    return hyps

def get_refs(conll_trained_data):
    refs = []
    for token in conll_trained_data:
        if(token != None): 
            properties = token.split()
            refs.append((properties[0],properties[-1]))
    #print(refs)
    return refs    

def print_possible_labels(conll_data):
    # spacy labels
    #print ('\n', nlp.get_pipe("parser").labels,'\n')
    #print (nlp.get_pipe("tagger").labels,'\n')
    print (nlp.get_pipe("ner").labels,'\n')

    # conll 2003 labels
    conll_labels = []   
    for conll_token in conll_data:
        properties = conll_token.split()
        if(properties[-1] not in conll_labels):
            conll_labels.append(properties[-1])
    print(conll_labels,'\n')

def simple_evaluation(refs,hyps):
    correct = 0
    incorrect = 0
    classes = []
    index = -1
    for ref,hyp in zip(refs,hyps): 
        index = -1
        for item in classes:
            if(ref[1] == item[0]):
                index = classes.index(item)
        if(index == -1):
            classes.append([ref[1],0,0])  
        
        if(ref[1] == hyp[1]):
            correct += 1
            classes[index][1] += 1
        else:
            incorrect +=1
            classes[index][2] += 1

    total_accuracy = correct/(correct+incorrect)
    print("Simple Evaluation Accuracy")
    print("Total:", total_accuracy)

    for ent_type in classes:        

        if(ent_type[1] == 0 and ent_type[2] == 0):
            print("Actually,",ent_type[0],"did not appear in the corpus")
        else: 
            print("Accuracy of",ent_type[0], ": ", (ent_type[1]/(ent_type[1]+ent_type[2])))
    return total_accuracy

# MAIN

nlp = spacy.load("en_core_web_sm")
conll_data = get_chunks("data/test.txt")
conll_data = set(filter(None, conll_data)) # removes None values from dataset

# Applying spacy named entity recognition to the trained data
corpus = tokenized_back_to_string(conll_data)
spacy_doc = nlp(corpus) # tokenize original reconstructed corpus but using spacy
# 1st step - reconstruction of the tokenization
spacy_doc = reconstruct_hyphenated_words(spacy_doc) # addressing the hyphen conversion issue
# 2nd step - adaptation to the proper format for evaluation 

# getting references for conll evaluation
refs = get_refs(conll_data)
# getting hypothesis for conll evaluation
hyps = get_hyps(spacy_doc)

#print_tokens_from_both_corpus_simultaneously(conll_data,spacy_doc)
#print_possible_labels(conll_data)
#print_processed_tokens_from_both_corpus_simultaneously(hyps,refs)

simple_results = simple_evaluation(refs,hyps)

#results = evaluate(refs, hyps)
#print(results)
print("\nConlleval evaluation outputs a strange error so we're skipping it, but feel welcome to test it! Here are the parameters(refs and hyps):\n")
print(refs,'\n')
print(hyps,'\n')

entity_grouping(spacy_doc)

extend_entity_span(spacy_doc)

# Reference code from class examples

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

#print([(token.text, token.ent_iob_, token.ent_type_, token.whitespace_)]) # text, beginning or end of sentence, entity type, if there's a whitespace after or not

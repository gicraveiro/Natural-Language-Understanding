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

import conll
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

def extract_tokens(list_of_tokenized_corpus):
    sentences = []
    i = 0

    for sent in list_of_tokenized_corpus:
        sentences.append([])
        for tokenized in sent:
            str_format_token = ''.join(tokenized)
            #print(str_format_token)
            aux = str_format_token.split() # transforms the string of the tokenized item into a list of strings ( each string is one component of the trained token)
            sentences[i].append(aux[0])
        #print(sentences[i])
        i = i + 1   
    #print(sentences)
    return sentences


# extracts token texts and return them in a format of a list of lists, each list containing all tokens of the sentence
def group_noun_chunks(corpus):
    sentences = [[]]
    i = 0
    
    for tokenized in corpus:
        if ( tokenized != None):
            aux = tokenized.split() # transforms the string of the tokenized item into a list of strings ( each string is one component of the trained token)
            #print(aux)
            if (len(aux) >= 3 and aux[2].startswith("B-")): # identifies the beginning of a new phrase
                if (len(sentences) == 1 and len(sentences[0]) == 0): # identifies if it is the first token from the first sentence that will be added to the list (initial token)
                    i = 0 # first sentence of the list position
                else:
                    i = i + 1 # indicates the position of the next sentence in the list
                    sentences.append([]) # creates the list for the coming sentence
                    #print(sentences[i-1])
                    #print("NEWLIST")
                    #print(aux[0],aux[2])
            sentences[i].append(aux[0]) # adds only the token text to the correct sentence in the list of sentences

    #for sent in sentences:   
    #    print (sent)
    print(sentences)
    return sentences

def tokenized_back_to_string(list_of_tokenized_corpus):

    #print(list_of_tokenized_corpus)
    list_of_lists_of_tokens = extract_tokens(list_of_tokenized_corpus)
    #list_of_lists_of_tokens = group_noun_chunks(data)  # one list containing many lists, that contain many strings, each one representing a token text
    list_of_sentences = [] # one list containing many strings, each one representing a sentence
    #print(list_of_lists_of_tokens)

    #Group tokens belonging to the same sentence into one string in the list of sentences
    for list in list_of_lists_of_tokens:
        str_format_sentence = ' '.join(list)
        list_of_sentences.append(str_format_sentence) 
    #print(list_of_sentences)

    # Groups the strings of all sentences into one single corpus string

    #original_corpus_rebuilt = ""
    #for str_format_sentence in list_of_sentences:
    #    original_corpus_rebuilt += str_format_sentence  # would group all string but not separate them with spaces so let's cancel this approach
        #print(original_corpus_rebuilt);
    original_corpus_rebuilt = ' '.join(list_of_sentences)
    #print(original_corpus_rebuilt)

    return original_corpus_rebuilt

# Reconstruct tokens
#   merge tokens that were separated by hyphens 
def reconstruct_hyphenated_words(corpus):
    i = 0
    #print([(t.text, t.ent_iob_, t.ent_type_, t.whitespace_) for t in corpus])
    while i < len(corpus):
        if(corpus[i].text == "-" and corpus[i].whitespace_ == ""): # identify hyphen ("-" inside a word)
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+2]) # merge the first part of the word, the hyphen and the second part of the word
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
    i = 0
    for conll_sentence in conll_data:
        for conll_token in conll_sentence:       
            properties_aux = conll_token[0] # "transforms" tuple format to string format
            properties = properties_aux.split()
            print(properties[0],' -> ',spacy_doc[i].text)
            print(properties[-1],' -> ',spacy_doc[i].ent_iob_ +'-'+spacy_doc[i].ent_type_+'\n')  
            i += 1
        if(properties[0] != spacy_doc[i-1].text): break

def reconstruct_spacy_tokenization(conll_data,spacy_doc):
    i = 0
    for conll_sentence in conll_data:
        for conll_token in conll_sentence:           
            properties_aux = conll_token[0] # "transforms" tuple format to string format
            properties = properties_aux.split()
            while (properties[0] != spacy_doc[i].text):
                with spacy_doc.retokenize() as retokenizer:
                    retokenizer.merge(spacy_doc[i:i+2])
            i += 1

def print_processed_tokens_from_both_corpus_simultaneously(hyps,refs):
    
    for ref,hyp in zip(refs,hyps):
        for token_ref,token_hyp in zip(ref,hyp):
            print(token_ref,token_hyp)

def get_hyps(conll_trained_data, spacy_doc):
    hyps = []
    i = 0
    j = 0
    for sent in conll_trained_data:
    #for sent in spacy_doc.sents:
        #print(sent)
        hyps.append([])
        for token in sent:
            ent_type = spacy_doc[j].ent_type_
            if(ent_type == 'PERSON'):
                ent_type = 'PER'
            elif(ent_type == 'O' or ent_type == 'LOC' or ent_type == 'ORG'):
                pass # doesnt change it
            else:
                ent_type = 'MISC'
        
            if(spacy_doc[j].ent_iob_ == "O"):
                iob = "O"
            else: 
                iob = (spacy_doc[j].ent_iob_ +'-'+ent_type)
            hyps[i].append((spacy_doc[j].text,iob))
            j = j + 1
        i = i + 1
        #j = 0
    #print(hyps)
    return hyps

def get_refs(conll_trained_data):
    #print(conll_trained_data)
    refs = []
    i = 0

    for sent in conll_trained_data:
        #print(sent)
        refs.append([])
        #refs = [[(text, iob) for text, pos, iob in sent] for sent in conll_trained_data]
        for token in sent:
            str = ''.join(token)
            #print(str)
            if(str != None): 
                properties = str.split()
                refs[i].append((properties[0],properties[-1]))
                
        #print(refs[i])
        i = i + 1        
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
conll_data_sents_list_format = read_corpus_conll("data/test.txt")
conll_data = get_chunks("data/test.txt")
conll_data = set(filter(None, conll_data)) # removes None values from dataset
#print(conll_data_sents_list_format)

# Applying spacy named entity recognition to the trained data

#corpus = tokenized_back_to_string(conll_data) # takes the trained data as input and outputs the original corpus (not trained)
corpus = tokenized_back_to_string(conll_data_sents_list_format)
#print(corpus[0:100]);
spacy_doc = nlp(corpus) # tokenize original reconstructed corpus but using spacy
#print(spacy_doc[0:100])
# 1st step - reconstruction of the tokenization
reconstruct_spacy_tokenization(conll_data_sents_list_format, spacy_doc)
#spacy_doc = reconstruct_hyphenated_words(spacy_doc) # addressing the hyphen conversion issue
# 2nd step - adaptation to the proper format for evaluation 
#print(corpus[0:500]);
#print(spacy_doc[0:500])
# getting references for conll evaluation
refs = get_refs(conll_data_sents_list_format)
#print(refs)
# getting hypothesis for conll evaluation
hyps = get_hyps(conll_data_sents_list_format, spacy_doc)
#print('\n\n\n')
#print(hyps)


#print_tokens_from_both_corpus_simultaneously(conll_data,spacy_doc)
#print_tokens_from_both_corpus_simultaneously(conll_data_sents_list_format,spacy_doc)
#print_tokens_from_both_corpus_simultaneously(conll_data,conll_data_sents_list_format)
#print_possible_labels(conll_data)
#print_processed_tokens_from_both_corpus_simultaneously(hyps,refs)

#print(refs[1])
#print(hyps[1])
#print(refs[-1])
#print(hyps[-1])
#simple_results = simple_evaluation(refs,hyps)

#results = evaluate(refs, hyps)
#print(results)
#print("\nConlleval evaluation outputs a strange error so we're skipping it, but feel welcome to test it! Here are the parameters(refs and hyps):\n")
#print(refs,'\n')
#print(hyps,'\n')

#entity_grouping(spacy_doc)

#extend_entity_span(spacy_doc)

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

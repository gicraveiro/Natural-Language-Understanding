'''

Assignment 1 - Natural Language Understanding Course - University of Trento

Author: Giovana Meloni Craveiro


Assignment: Working with Dependency Graphs (Parses)

The objective of the assignment is to learn how to work with dependency graphs by defining functions.

Read spaCy documentation on dependency parser to learn provided methods.

Define functions to:

    expract a path of dependency relations from the ROOT to a token
    extract subtree of a dependents given a token
    check if a given list of tokens (segment of a sentence) forms a subtree
    identify head of a span, given its tokens
    extract sentence subject, direct object and indirect object spans

'''
import spacy
from spacy.lang.en import English

def parser(sentence):
    doc = spacy_nlp(sentence)
    print('Example sentence:',doc,'\n')
    return doc

# extract a path of dependency relations from the ROOT to a token
def extract_path_root_token(sentence): 

    sentences = parser(sentence)
    for token in sentences:
        print("Path from root",sentences[0:-1].root, "to token",token.text)
        for ancestor in sorted(token.ancestors):
            print("{}\t{}".format(ancestor.text,ancestor.dep_))
        print("{}\t{}".format(token.text,token.dep_))
        print(' ')

# extract subtree of a dependents given a token
def extract_subtree(sentence):

    doc = parser(sentence)

    for sent in doc.sents:
        for token in sent:
            print(token.text,"subtree:")
            for descendant in token.subtree:
                print(descendant)
            print(' ')
            
    print(' ')
    return token.subtree

# check if a given list of tokens (segment of a sentence) forms a subtree
def check_tokens_form_subtree(sentence, segment):
    doc = parser(sentence)
    seg = parser(segment)
    result = "False"   
    seg_list = []
    subtree = []

    for token in seg:
        seg_list.append(token.text)
    
    for sent in doc.sents:
        for token in sent:
            for i in range(len(subtree)):
                subtree.remove(subtree[0])
            for descendant in token.subtree:
                subtree.append(descendant.text)

            if (subtree == seg_list):
                result = "True"
    print("Result:",result,'\n')
    return result

# identify head of a span, given its tokens
def identify_head(sequence):

    doc = parser(sequence)
    for token in doc:
        if (token.head == token):
            root = token
    #root = [token for token in doc if token.head == token][0] # alternative way: list comprehension -> creates a list that contains only the root
    return root

# extract sentence subject, direct object and indirect object spans
def extract_constituents(sentence): 

    doc = parser(sentence)
    span = doc[0:len(doc)]
    constituents = set()
    subject, dobj, iobj = [],[],[]
    for token in span:
        print(token.text)
        if token.dep_ == "nsubj": 
            for descendant in token.subtree:
                subject.append(descendant.text)

        elif token.dep_ == "dobj":
            for descendant in token.subtree:
                dobj.append(descendant.text)

        elif token.dep_ == "dative": 
            for descendant in token.subtree:
                iobj.append(descendant.text)
    
    print("Subject:", (' '.join(subject) if subject else 'None'))
    print("Direct object:", (' '.join(dobj) if dobj else 'None'))
    print("Indirect object: ", (' '.join(iobj) if iobj else 'None'))  

    return None

spacy_nlp = spacy.load('en_core_web_sm')

example = 'Lola and Anna gave me a birthday cake'
sequence = 'difficult tasks'
segment = 'with a telescope'

print("Function 1 - Extract path of dependency relations from root to token\n")
extract_path_root_token(example)

print("\nFunction 2 - Subtrees of each token in the sentence:")
extract_subtree(example)

print("Function 3 - Check if a given list of tokens forms a subtree\n")
check_tokens_form_subtree(example, segment)

print('\nFunction 4 - Identify head of the span\n')
print( 'ROOT:',identify_head(sequence),'\n')

print('\nFunction 5 - Extract constituents\n')
extract_constituents(example)

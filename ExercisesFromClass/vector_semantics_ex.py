

#Exercise
#Extend the co-occurence matrix computation to allow specifying window of context (as tuple for previous and next words)
#Define a funtion to compute PPMI on co-occurence matrix (Optional)

# 1 - 
import numpy as np

def get_vocab(samples):
    vocab = set()
    for s in samples:
        words = s if type(s) is list else s.split() # transforms sentence string in list of words if needed
        vocab = vocab.union(set(words)) # adds only new words to the existing vocabulary, which is a set
    return sorted(list(vocab)) #after set of the vocabulary is ready, transform it into an alphabetical list

def cooc_matrix_context_window(samples, vocab, start, end):
    m = np.zeros((len(vocab), len(vocab)))
    # let's co-occurence be document level (i.e. sentence)
    for s in samples: # list of words of the sentence
        s = s[start:end]
        for w1 in s:  # rows
            for w2 in s:  # columns
                i = vocab.index(w1)
                j = vocab.index(w2)
                m[i][j] += 1 
    return m

data = [
    "the capital of France is Paris", 
    "Rome is the capital of Italy",
]

vocab = get_vocab(data)
print(vocab)
cm = cooc_matrix_context_window([s.split() for s in data], vocab, 1,4)
print(cm)

#Define a funtion to compute PPMI on co-occurence matrix (Optional)
# 2 -









'''

Exercises
Implement one-hot encoding (binary vecorization)
takes vocabulary and a sentence as arguments (lists of words)
outputs numpy vector (ndarray)
Implement a function to compute cosine similarity using numpy methods
np.dot
np.sqrt
Using the defined functions
vectorize the sentences:
"the capital of France is Paris"
"Rome is the capital of Italy"
compute cosine similarity between them
compare similarity values to the cosine similarity using the ouput of (scipy.spatial.distance.cosine)
i.e. use distance to compute similarity

'''






'''
Exercise
Implement analogy compuatation function that

takes 3 words of the analogy task as an input
outputs the word 4
make use of spacy vectors
version 2:

implement without using spacy's most_similar

'''
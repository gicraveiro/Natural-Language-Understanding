#import os
#import sys
#sys.path.insert(0, os.path.abspath('../src/'))

#nlp = spacy.load('conll2002')
nltk.download('conll2002')

from nltk.corpus import conll2002

print(len(conll2002.tagged_sents()))
print(conll2002._chunk_types)
print(conll2002.sents('esp.train')[0])
print(conll2002.tagged_sents('esp.train')[0])
print(conll2002.chunked_sents('esp.train')[0])
print(conll2002.iob_sents('esp.train')[0])

#print(conll2002)
#print(conll2002.__file__)
#from nltk import conll

#from conll import evaluate



# training hmm on training data: exactly as above
import nltk.tag.hmm as hmm

hmm_model = hmm.HiddenMarkovModelTrainer()

print(conll2002.iob_sents('esp.train')[0])

# let's get only word and iob-tag
trn_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
print(trn_sents[0])

tst_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]

hmm_ner = hmm_model.train(trn_sents)
    
# evaluation
accuracy = hmm_ner.evaluate(tst_sents)

print("Accuracy: {:6.4f}".format(accuracy))



# to import conll
import os
import sys
sys.path.insert(0, os.path.abspath('../data/'))

from conll import evaluate
# for nice tables
import pandas as pd

# getting references (note that it is testb this time)
refs = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testb')]
print(refs[0])
# getting hypotheses
hyps = [hmm_ner.tag(s) for s in conll2002.sents('esp.testb')]
print(hyps[0])

results = evaluate(refs, hyps)

pd_tbl = pd.DataFrame().from_dict(results, orient='index')
pd_tbl.round(decimals=3)


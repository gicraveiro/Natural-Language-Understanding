from sklearn.datasets import load_iris
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

data = load_iris()

print("Classes: {}".format(len(list(data.target_names))))
print("Samples: {}".format(len(data.data)))
print("Dimensionality: {}".format(len(list(data.feature_names))))
print("Samples per Class: {}".format(dict(Counter(list(data.target)))))

print(data.data[0])  # prints feature vector

print(data.data.shape)  # prints matrix shape for data
print(data.target.shape)  # print matrix shape for labels

print(data.DESCR)  # prints full data set description
print(data.data)  # prints features
print(data.target) # prints labels

# Random K-fold split
random_split = KFold(n_splits=5, shuffle=True)
print("Random K-fold split")

for train_index, test_index in random_split.split(data.data):
    
    print("Samples per Class in Training: {}".format(dict(Counter(list(data.target[train_index])))))
    print("Samples per Class in Testing: {}".format(dict(Counter(list(data.target[test_index])))))

#Stratified K-fold split
print("Stratified K-fold split")
stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

for train_index, test_index in stratified_split.split(data.data, data.target):
    
    print("Samples per Class in Training: {}".format(dict(Counter(list(data.target[train_index])))))
    print("Samples per Class in Testing: {}".format(dict(Counter(list(data.target[test_index])))))

from sklearn.naive_bayes import GaussianNB

# choose classification algorithm & initialize it
clf = GaussianNB()

# for each training/testing fold
for train_index, test_index in stratified_split.split(data.data, data.target):
    # train (fit) model
    clf.fit(data.data[train_index], data.target[train_index])
    # predict test labels
    clf.predict(data.data[test_index])
    # score the model (using average accuracy for now)
    accuracy = clf.score(data.data[test_index], data.target[test_index])
    print("Accuracy: {:.3}".format(accuracy))

from sklearn.dummy import DummyClassifier

random_clf = DummyClassifier(strategy="uniform")

for train_index, test_index in stratified_split.split(data.data, data.target):
    random_clf.fit(data.data[train_index], data.target[train_index])
    random_clf.predict(data.data[test_index])
    accuracy = random_clf.score(data.data[test_index], data.target[test_index])
    
    print("Uniform Accuracy: {:.3}".format(accuracy))

random_clf = DummyClassifier(strategy="stratified")

for train_index, test_index in stratified_split.split(data.data, data.target):
    random_clf.fit(data.data[train_index], data.target[train_index])
    random_clf.predict(data.data[test_index])
    accuracy = random_clf.score(data.data[test_index], data.target[test_index])
    
    print("Stratified Accuracy: {:.3}".format(accuracy))

random_clf = DummyClassifier(strategy="most_frequent")

for train_index, test_index in stratified_split.split(data.data, data.target):
    random_clf.fit(data.data[train_index], data.target[train_index])
    random_clf.predict(data.data[test_index])
    accuracy = random_clf.score(data.data[test_index], data.target[test_index])
    
    print("Most-frequent Accuracy: {:.3}".format(accuracy))

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# choose classification algorithm & initialize it
clf = GaussianNB()

# for each training/testing fold
for train_index, test_index in stratified_split.split(data.data, data.target):
    # train (fit) model
    clf.fit(data.data[train_index], data.target[train_index])
    # predict test labels
    hyps = clf.predict(data.data[test_index])
    refs = data.target[test_index]
    
    report = classification_report(refs, hyps, target_names=data.target_names)
    
    print(report)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# choose classification algorithm & initialize it
clf = GaussianNB()
# get scores
scores = cross_val_score(clf, data.data, data.target, cv=5)

print(scores)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

# choose classification algorithm & initialize it
clf = GaussianNB()
# scoring providing our custom split & scoring using 
scores = cross_validate(clf, data.data, data.target, cv=stratified_split, scoring=['f1_macro'])

print(sum(scores['test_f1_macro'])/len(scores['test_f1_macro']))

# try different evaluation scores
scores2 = cross_validate(clf, data.data, data.target, cv=stratified_split, scoring=['f1_micro'])

scores3 = cross_validate(clf, data.data, data.target, cv=stratified_split, scoring=['f1_weighted'])
print('Scores Micro')
print( sum(scores2['test_f1_micro'])/len(scores2['test_f1_micro']))
print('Scores weighted')
print(sum(scores3['test_f1_weighted'])/len(scores3['test_f1_weighted']))

#Vectorization

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'who plays luke on star wars new hope',
    'show credits for the godfather',
    'who was the main actor in the exorcist',
    'find the female actress from the movie she \'s the man',
    'who played dory on finding nemo'
]

vectorizer = CountVectorizer()

# use fit_transform to 'learn' the features and vectorize the data
vectors = vectorizer.fit_transform(corpus)

print(vectors.toarray())  # print numpy vectors



test_corpus = [
    'who was the female lead in resident evil',
    'who played guido in life is beautiful'
]

# 'trained' vectorizer can be later used to transform the test set 
test_vectors = vectorizer.transform(test_corpus)
print(test_vectors.toarray())

# Exercise
# Using Newsgroup dataset from scikit-learn train and evaluate Multinomial Naive Bayes model Experiment with different vectorization methods and parameters:

#- `binary` of Count Vecrorization
#- TF-IDF Transformation
#- min and max cut-offs
#- using stop-words
#- lowercasing









#final part that didnt have an explanation
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

vectorizer = CountVectorizer()
classifier = MultinomialNB()

trn_vectors = vectorizer.fit_transform(newsgroups_train.data)
tst_vectors = vectorizer.transform(newsgroups_test.data)

classifier.fit(trn_vectors, newsgroups_train.target)
predictions = classifier.predict(tst_vectors)

print(classification_report(newsgroups_test.target, predictions, target_names=newsgroups_train.target_names))





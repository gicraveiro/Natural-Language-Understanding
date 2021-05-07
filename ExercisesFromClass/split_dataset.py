from sklearn.model_selection import KFold

random_split = KFold(n_splits=5, shuffle=True)

for train_index, test_index in random_split.split(data.data):
    
    print("Samples per Class in Training: {}".format(dict(Counter(list(data.target[train_index])))))
    print("Samples per Class in Testing: {}".format(dict(Counter(list(data.target[test_index])))))
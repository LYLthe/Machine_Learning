import operator
from numpy import *


def createdataset():

    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def calssify(inx, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inx, (datasetsize, 1))-dataset
    sqdiffmat = diffmat*2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances**0.5
    sorted_distenaces = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distenaces[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


dattaset, labelss = createdataset()
p = calssify([0, 0], dattaset, labelss, 3)
print(p)

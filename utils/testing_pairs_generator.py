# generate the testing paris (.txt file)
# half of them are positive pairs, half are negative pairs

import os
import collections
import numpy as np
from glob import glob

dataset = '/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household/patches'
num_pairs = 5000
labels = []

def load_labels(dataset):
    with open(os.path.join(dataset, 'info.txt'), 'r') as f:
        for line in f:
            labels.append(int(line.split()[0]))
    return np.asarray(labels)

def create_indices(labels):
    old = labels[0]
    indices = dict()
    indices[old] = 0
    for x in range(len(labels) - 1):
        new = labels[x+1]
        if old != new:
            indices[new] = x+1
        old = new
    return indices

def main(dataset):
    labels = load_labels(dataset)
    count = collections.Counter(labels)
    indices = create_indices(labels)
    unique_labels = np.unique(labels)
    n_classes = unique_labels.shape[0]
    # add only unique
    already_idxs = set()
    num_p = np.min((int(num_pairs/2), n_classes))
    print('number of the positive pairs: %s' % num_p)
    pospair, negpair=[], []
    # positive pairs
    for _ in range(num_p):
        c = np.random.randint(0, n_classes)  # in [0, n_classes)
        while c in already_idxs or count[c] < 2:
            c = np.random.randint(0, n_classes)
        already_idxs.add(c)
        if count[c] == 2:
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, count[c])
            n2 = np.random.randint(0, count[c])
            while n1 == n2:
                n2 = np.random.randint(0, count[c])
        pospair.append([indices[c]+n1, c, 0, indices[c]+n2, c, 0])
    # negative pairs
    already_idxs = set()
    for _ in range(num_p):
        c1 = np.random.randint(0, n_classes)
        while c1 in already_idxs:
            c1 = np.random.randint(0, n_classes)
        already_idxs.add(c1)
        c2 = np.random.randint(0, n_classes)
        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        n1 = np.random.randint(0, count[c1])
        n2 = np.random.randint(0, count[c2])
        negpair.append([indices[c1]+n1, c1, 0, indices[c2]+n2, c2, 0])
    matches = []
    matches.extend(pospair)
    matches.extend(negpair)
    np.random.shuffle(matches)
    matches = np.asarray(matches)
    # np.savetxt(os.path.join(dataset, 'm50_5000_5000_0.txt'), matches, delimiter=' ')
    with open(os.path.join(dataset, 'm50_5000_5000_0.txt'), 'w') as f:
        f.write('# indice1 class1 0 indice2 class2 0')
        for line in matches:
            one = [str(ele) for ele in line]
            f.write('\n'+" ".join(one))

if __name__ == '__main__':
    main(dataset)
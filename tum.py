import sys
import os

import random
import collections
from tqdm import tqdm

import cv2
import numpy as np
from preprocess import normalize_depth

class TUMDataset(object):
    # the extension of the images containing the patches
    # IMAGE_EXT = 'dep.png'

    # the size of the patches once extracted
    PATCH_SIZE = 64
    # the number of patches per row/column in the
    # image containing all the patches
    PATCHES_PER_ROW = 16
    PATCHES_PER_IMAGE = PATCHES_PER_ROW ** 2

    def __init__(self, base_dir, params, test=False):
        # check that the directories exist
        assert os.path.isdir(base_dir) == True, \
            "The given directory doesn't exist: %s" % base_dir

        # the dataset base directory
        self._base_dir = base_dir

        self.num_channels = params['num_channels']
        if params['patch_type'] == 'gray':
            self.IMAGE_EXT = 'rgb.png'
        elif params['patch_type'] == 'rgb':
            self.IMAGE_EXT = 'rgb.png'
        elif params['patch_type'] == 'dep':
            self.IMAGE_EXT = 'dep.png'
        else:
            print("Invalid patch type!")

        # testing variables
        self.test = test
        self.n = 128

        # the loaded patches
        self._data = dict()

    def get_batch(self, data, data_ids, step, batch_size):
        # compute the offset to get the correct batch
        offset = step * batch_size % len(data_ids)
        # get a triplet batch from the training data
        ids = data_ids[offset:offset + batch_size]

        a, p, n = [[] for _ in range(3)]
        for id in ids:
            a.append(data[id[0]])
            p.append(data[id[1]])
            n.append(data[id[2]])
        return a, p, n

    def load_by_name(self, type, names, params):
        # assert name in self._data.keys(), \
        #     "Dataset doesn't exist: %s" % name
        if type == "training":
            maxNum = params['max_walls']
        elif type == "testing":
            maxNum = 100000000
        else:
            maxNum = 0
            print("Invalid type!")

        img_file_all = []
        patches_all = []
        labels_all = [-1]
        self._data[type] = dict()
        for name in names:
            self._data[type][name] = dict()
            assert os.path.exists(os.path.join(self._base_dir, name)) == True, \
                "The dataset directory doesn't exist: %s" % name
            # check if the dataset is already loaded
            # if self._data[name] is not None:
            #     print ('[INFO] Dataset is cached: %s' % name)
            #     return
            # load the images containing the patches
            img_file = self._load_image_fnames(self._base_dir, name)
            img_file_all.extend(img_file)
            labels = self._load_labels(self._base_dir, name)
            # load the dataset ground truth matches only for testing
            if type == "testing":
                matches = self._load_matches(self._base_dir, name)
                self._data[type]['matches'] = matches
            if len(img_file_all) > maxNum:
                patches = self._load_patches(img_file[:-(len(img_file_all)-maxNum)], params)
                labels = labels[:len(patches)]
                self._data[type][name]['patches'] = patches
                self._data[type][name]['labels'] = labels
                break
            patches = self._load_patches(img_file, params)
            self._data[type][name]['patches'] = patches
            self._data[type][name]['labels'] = labels
        print('The dataset used for {}'.format(type))

        for name in names:
            if name in self._data[type]:
                print ('Dataset|NumOfPatch|NumOfLabels:  %s | %s | %s' %
                       (name, len(self._data[type][name]['patches']), len(self._data[type][name]['labels'])))
                patches_all.extend(self._data[type][name]['patches'])
                labels_all.extend(np.asarray(self._data[type][name]['labels']) + labels_all[-1] + 1)
        patches_all = np.asarray(patches_all, np.float32)

        #==================preprocessing for depth patches===============
        if params['patch_type'] == 'dep':
            patches_all = normalize_depth(patches_all)
            # patches_cv = patches_all[:,16,16,0]   # center value of each patch
            # patches_all = patches_all - np.tile(np.reshape(patches_cv, (-1, 1, 1, 1)), (1, 32, 32, 1))
            # patches_all = np.clip(patches_all, -500, 500)
            # patches_all = patches_all / 500.

        self._data[type]['patches'] = patches_all
        self._data[type]['labels'] = np.asarray(labels_all[1:])    # 1: to drop the first element in label list
        print('-- Number of patches: %s' % len(self._data[type]['patches']))
        print('-- Number of labels:  %s' % len(self._data[type]['labels']))
        print('-- Number of ulabels: %s' % len(np.unique(self._data[type]['labels'])))

    def generate_triplets(self, params):
        n_triplets = params['n_triplets']
        batch_size = params['batch_size']
        # retrieve loaded patches and labels
        labels = self._get_labels()
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self._create_indices(labels)
        # unique label
        unique_labels = np.unique(labels)
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()
        # range for the sampling
        labels_size = len(labels) - 1
        # triplets ids
        triplets = []
        # generate the triplets
        for x in tqdm(range(n_triplets)):
            if len(already_idxs) >= batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs or count[c1]<2:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes - 1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if count[c1] == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, count[c1])
                n2 = np.random.randint(0, count[c1])
                while n1 == n2:
                    n2 = np.random.randint(0, count[c1])
            n3 = np.random.randint(0, count[c2])
            triplets.append([indices[c1]+n1, indices[c1]+n2, indices[c2]+n3])
        return np.array(triplets)

    def generate_stats(self, name):
        print ('-- Computing dataset mean: %s ...' % name)
        # compute the mean and std of all patches
        patches = self._get_patches(name)
        mean, std = self._compute_mean_and_std(patches)
        print ('-- Computing dataset mean: %s ... OK' % name)
        print ('-- Mean: %s' % mean)
        print ('-- Std : %s' % std)
        return mean, std

    def prune(self, name, min=2):
        labels = self._get_labels(name)
        # filter the labels
        ids, labels = self._prune(labels, min)
        # return only the filtered patches
        return ids, labels

    def _prune(self, labels, min):
        # count the number of labels
        c = collections.Counter(labels)
        # create a list with globals indices
        ids = range(len(labels))
        # remove ocurrences
        ids, labels = self._rename_and_prune(labels, ids, c, min)
        return np.asarray(ids), np.asarray(labels)

    def _rename_and_prune(self, labels, ids, c, min):
        count, x = 0, 0
        labels_new, ids_new = [[] for _ in range(2)]
        while x < len(labels):
            num = c[labels[x]]
            if num >= min:
                for i in range(num):
                    labels_new.append(count)
                    ids_new.append(ids[x + i])
                count += 1
            x += num
        return ids_new, labels_new

    def _load_matches(self, base_dir, name):
        """
        Return a list containing the ground truth matches
        """
        fname = os.path.join(base_dir, name, 'patches', 'm50_5000_5000_0.txt')
        assert os.path.isfile(fname), 'Not a file: %s' % fname
        # read file and keep only 3D point ID and 1 if is the same, otherwise 0
        matches = []
        with open(fname, 'r') as f:
            next(f)
            for line in f:
                l = line.split()
                matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])
        return np.asarray(matches)

    def _load_image_fnames(self, base_dir, dir_name):
        """
        Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        dataset_dir = os.path.join(base_dir, dir_name, 'patches')
        for file in os.listdir(dataset_dir):
            if file.split('_')[-1] == self.IMAGE_EXT :
                files.append(os.path.join(dataset_dir, file))
        return sorted(files)  # sort files in ascend order to keep relations

    def _load_patches(self, img_files, params):
        """
        Return a list containing all the patches
        """
        patch_size = params['patch_size']
        num_channels = params['num_channels']
        patches_all = []
        # reduce the number of files to load if we are in testing mode
        img_files = img_files[0:self.n] if self.test else img_files
        # load patches
        pbar = tqdm(img_files)
        for file in pbar:
            name = file.split('/')[-3]
            pbar.set_description('Loading dataset %s' % name)
            # pick file name
            assert os.path.isfile(file), 'Not a file: %s' % file
            # load the image containing the patches and convert to float point
            # and make sure that que use only one single channel
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)# / 255.
            if (params['patch_type'] == 'rgb'):
                # img = cv2.imread(file, cv2.IMREAD_UNCHANGED) / 255.
                img = img / 255.
            elif (params['patch_type'] == 'gray'):
                # img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.
            elif (params['patch_type'] == 'dep'):
                # img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                img = img

            # split the image into patches and
            # add patches to buffer as individual elements
            patches_row = np.split(img, self.PATCHES_PER_ROW, axis=0)
            for row in patches_row:
                patches = np.split(row, self.PATCHES_PER_ROW, axis=1)
                for patch in patches:
                    # resize the patch
                    patch_resize = cv2.resize(patch, (patch_size, patch_size))
                    # convert to tensor [w x h x d]
                    patch_tensor = patch_resize.reshape(patch_size,
                                                        patch_size,
                                                        num_channels)
                    patches_all.append(patch_tensor)
        # return np.asarray(patches_all) if not self.test \
        #     else np.asarray(patches_all[0:self.n])
        return patches_all

    def _load_labels(self, base_dir, dir_name):
        """
        Return a list containing all the labels for each patch
        """
        info_fname = os.path.join(base_dir, dir_name, 'patches', 'info.txt')
        assert os.path.isfile(info_fname), 'Not a file: %s' % info_fname
        # read file and keep only 3D point ID
        labels = []
        with open(info_fname, 'r') as f:
            for line in f:
                labels.append(int(line.split()[0]))
        # return np.asarray(labels) if not self.test \
        #     else np.asarray(labels[0:self.n])
        return labels

    def _create_indices(self, labels):
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in range(len(labels) - 1):
            new = labels[x + 1]
            if old != new:
                indices[new] = x + 1
            old = new
        return indices

    def _compute_mean_and_std(self, patches):
        """
        Return the mean and the std given a set of patches.
        """
        assert len(patches) > 0, 'Patches list is empty!'
        # compute the mean
        mean = np.mean(patches)
        # compute the standard deviation
        std = np.std(patches)
        return mean, std

    def _get_data(self, name):
        assert self._data[name] is not None, 'Dataset not loaded: %s' % name
        return self._data[name]

    def _get_patches(self, type):
        return self._data[type]['patches']

    def _get_matches(self):
        # return self._get_data(name)['matches']
        assert self._data['testing'] is not None, 'Testing dataset not loaded.'
        return self._data['testing']['matches']

    def _get_labels(self):
        return self._data['training']['labels']
import os

import glob2 as glob
from sklearn.model_selection import train_test_split


class Dataset():

    def __init__(self):
        self.object_types = {}

        self.datasetPath = None
        self.dataset = self.datasetPath

        self.divide_train_test = True

        # Set local paths to save data
        self.dictionary = None
        self.proj = None
        self.encodedImages = None
        self.labelsImages = None
        self.trained_network_path = None

        # Set the number of visual words for the dictionary
        self.numberOfVisualWords = 60
        self.numMaxDescriptor = 60 * 800

    def get_labels(self, pathString):
        pass

    def divide_train_test(self):
        pass


class PlushDataset(Dataset):

    def __init__(self):
        Dataset.__init__(self)

        self.object_types = {"banana": 0,
                             "watermelon": 1,
                             "mushroom": 2,
                             "zucchini": 3}

        self.datasetPath = "data/all/"
        self.datasetPath = "/media/luiza/559B56736E99E3AD/dati/luiza/201610-PlushObjects/all/"

        self.files = glob.glob(self.datasetPath + "/**/*.jpg") + glob.glob(self.datasetPath + "/**/*.png")
        self.dataset = self.datasetPath
        self.divide_train_test = False

        self.dictionary = "/tmp/visual_dictionary_plush"
        self.proj = "/tmp/pca_projection_plush"
        self.encodedImages = "/tmp/data_plush"
        self.labelsImages = "/tmp/labels_plush"
        self.trained_network_path = "/tmp/trained_gwr_plush"

        self.numberOfVisualWords = 60
        self.numMaxDescriptor = 60 * 800

    def get_labels(self, pathString):
        return self.object_types[os.path.basename(os.path.dirname(pathString)).lower()]

    def divide_train_test(self):
        # implement your own method

        print("####### divide train/test #######")
        bananas = glob.glob(self.datasetPath + "Banana/*.png")
        b, b_t, bl, bl_t = train_test_split(bananas, [self.object_types['banana'] for i in xrange(len(bananas))])

        mushroom = glob.glob(self.datasetPath + "Mushroom/*.png")
        m, m_t, ml, ml_t = train_test_split(mushroom, [self.object_types['mushroom'] for i in xrange(len(mushroom))])

        watermelon = glob.glob(self.datasetPath + "Watermelon/*.png")
        w, w_t, wl, wl_t = train_test_split(watermelon,
                                            [self.object_types['watermelon'] for i in xrange(len(watermelon))])

        zucchini = glob.glob(self.datasetPath + "Zucchini/*.png")
        z, z_t, zl, zl_t = train_test_split(zucchini, [self.object_types['zucchini'] for i in xrange(len(zucchini))])

        files = b + m + w + z
        labels = bl + ml + wl + zl

        files_test = b_t + m_t + w_t + z_t
        labels_test = bl_t + ml_t + wl_t + zl_t

        return files, labels, files_test, labels_test

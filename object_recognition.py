import os
import numpy as np
import cPickle as pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from VLADlib.VLAD import encode, get_descriptors, compute_visual_dictionary, encode_one_image
from dataset import PlushDataset
from growing_network.GrowingWhenRequired import GWR
from growing_network.utils import pca


def train_gwr(train_im,train_lbl, test_im, test_lbl, dictionary_file_name,projection_file_name,gwr_file_name,color='gray'):

    if not os.path.isfile(dictionary_file_name + '.pkl') or not os.path.isfile(projection_file_name + '.npy'):
        raise Exception('Visual dictionary and pca projection matrix not found!')
    else:
        visualDictionary = pickle.load(open(dictionary_file_name + '.pkl', "rb"))
        projections = np.load(projection_file_name + '.npy')

    print("####### encoding training dataset #######")
    data, images = encode(train_im, visualDictionary, projections, threads=8, color=color)


    print("####### encoding testing dataset #######")
    data_test, _ = encode(test_im, visualDictionary, projections, threads=8, color=color)
    X = np.asarray(data)
    Y = np.asarray(train_lbl)
    X_test = np.asarray(data_test)
    Y_test = np.asarray(test_lbl)

    if not os.path.isfile(gwr_file_name + '.p'):
        print '####### Training the GWR network #######'
        gwr = GWR()
        gwr.initialise_network(habituation_threshold=0.1, insertion_threshold=0.95, max_nodes=15000, max_age=100)
        gwr.train(x=X, y=Y, max_epoch=50, random_initialization=False)
        print '####### Saving the GWR network #######'
        gwr.save_network(gwr_file_name)
    else:
        gwr = GWR.load_network(gwr_file_name)

    print '####### Testing the GWR #######'
    _, predicted_labels, _ = gwr.classify(X_test)
    accuracy = accuracy_score(Y_test, predicted_labels)

    print('Classification accuracy on the test set: ' + str(accuracy))
    print('******************************************')
    print((classification_report(Y_test, predicted_labels)))
    print('******************************************')
    print(confusion_matrix(Y_test, predicted_labels, labels=None))
    print('******************************************')


def create_vlad_encoder(files, dictionary_file_name,projection_file_name,numMaxDescriptor,numberOfVisualWords,color='gray',threads = 8):

    if not os.path.isfile(dictionary_file_name + '.pkl'):

        print ("####### Extracting the descriptors #######")
        descriptors = get_descriptors(files, threads, numMaxDescriptor=numMaxDescriptor, color=color)
        print descriptors.shape

        print ("####### Performing PCA #######")
        descriptors, _, projection = pca(descriptors, 5)

        print ("####### Computing the visual dictionary with kMeans #######")
        visualDictionary = compute_visual_dictionary(descriptors, numberOfVisualWords)

        print ("####### Saving dictionary #######")
        np.save(projection_file_name, projection)
        with open(dictionary_file_name + '.pkl', 'wb') as f:
            pickle.dump(visualDictionary, f, protocol=pickle.HIGHEST_PROTOCOL)

    print "Done."


def gwrRecognition(dataset, imagePath):

    if not os.path.isfile(dataset.dictionary + '.pkl') or not os.path.isfile(dataset.proj + '.npy'):
        raise Exception('Visual dictionary and pca projection matrix not found!')
    else:
        visualDictionary = pickle.load(open(dataset.dictionary + '.pkl', "rb"))
        projections = np.load(dataset.proj + '.npy')

    if not os.path.isfile(dataset.trained_network_path + '.p'):
        raise Exception('Trained GWR not found!')
    else:
        gwr = GWR.load_network(dataset.trained_network_path)

    descs, _ = encode_one_image((imagePath,  visualDictionary,  projections))
    _, pred_label, _ = gwr.classify(descs.reshape(1, len(descs)))

    for object_name, object_id in  dataset.object_types.items():
        if object_id == pred_label[0]:
            print(object_name)

    return pred_label[0]


if __name__ == '__main__':

    import mnist

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    dictionary = '/tmp/mnist_dictionary'
    proj = '/tmp/mnist_projection'
    gwr_file_name = '/tmp/mnist_gwr'
    numberOfVisualWords = 60
    numMaxDescriptor = 60 * 8000

    # dataset = PlushDataset()
    create_vlad_encoder(train_images,dictionary,proj,numMaxDescriptor,numberOfVisualWords,color='gray')
    train_gwr(train_images, train_labels, test_images, test_labels, dictionary, proj, gwr_file_name, color='gray')
    #
    # gwrRecognition(dataset, '/tmp/zucchini2.png')
# Code adapted from: https://github.com/jorjasso/VLAD/blob/master/VLADlib/VLAD.py

import itertools
from sklearn.cluster import KMeans
import numpy as np
import glob2 as glob
from tqdm import tqdm
import multiprocessing
from imageio import imread
from VLADlib.vl_phow import phow


def get_k_descriptors(args):
    """
    Get a fixed number of PHOW features from one image.

    ..note : PHOW features are denseSIFT at different resolutions, hence a great number of features
    can be extracted from a dataset. We control this by picking k random features from each image.

    :param args:
    :return:
    """
    image, k,color=args
    try:
        im = imread(image) if isinstance(image, basestring) else image
        frames, descriptors = phow(im, color=color)

        if descriptors is not None:
            if 0 < k < len(descriptors):
                idx = np.arange(0, len(descriptors))[~np.all(descriptors == 0, axis=1)]
                if len(idx) > k:
                    np.random.shuffle(idx)
                    descriptors = descriptors[idx[:k]]
    except :
        print "ignore image " + image
        return None

    return descriptors


def get_descriptors(images, threads, color='gray', numMaxDescriptor=-1):
    """
    Extract descriptors from a list of paths or a list of images

    :param images: list of images or list of paths (strings) to images
    :param threads:
    :param numMaxDescriptor:
    :return:
    """

    if numMaxDescriptor > 0:
        numMaxDescriptor = int(np.ceil(numMaxDescriptor / len(images)))

    print('Found {} images...'.format(len(images)))
    print('Running with {} threads'.format(threads))

    if threads == 1:
        descriptors = []
        for imagePath in tqdm(images, desc="Extracting descriptors"):
            kp, des = get_k_descriptors((imagePath, numMaxDescriptor,color))
            if des is not None:
                descriptors.append(des)

        # flatten list
        descriptors = list(itertools.chain.from_iterable(descriptors))
    else:
        pool = multiprocessing.Pool(threads)

        data = [(im, numMaxDescriptor,color) for im in images] # Generate payload to send to threads
        descriptors = []
        for descs in tqdm(pool.imap_unordered(get_k_descriptors, data),
                          desc="[{} CPUs] Extracting descriptors".format(threads), total=len(data)):
            if descs is not None:
                descriptors.extend(descs)

        pool.close()

    #list to array
    descriptors = np.asarray(descriptors)

    return descriptors


def compute_visual_dictionary(training, k):
    """
    Compute a visual dictionary with kMeans
    :param training:  set of descriptors
    :param k: number of visual words
    :return: est
    :rtype: kMeans class
    """

    est = KMeans(n_clusters=k, init='k-means++', tol=0.0001, verbose=0).fit(training)
    return est


def encode_one_image(args):
    """
    Encode one image with VLAD.

    :param args: imagePath, visualDictionary, projection
    :return:
    """
    imagePath, visualDictionary, projection, color = args

    im = imread(imagePath)
    kp, des = phow(im, color=color)

    if des is not None:
        if projection is not None:
            des = np.dot(des, projection)

        v = improvedVLAD(des,visualDictionary)
        return v, imagePath

    return None, imagePath


def encode(files, visualDictionary, projection=None, threads = 1, color='gray'):
    """
    Encode the whole dataset with VLAD

    :param files:
    :param visualDictionary: a visual dictionary from k-means algorithm
    :param projection: pca projection matrix
    :param threads:
    :return:
    """
    descriptors_vlad=list()
    idImage =list()

    if threads == 1:
        for imagePath in tqdm(files, desc="Extracting VLAD features"):
            descrs,_ = encode_one_image((imagePath, visualDictionary, projection))
            if descrs is not None:
                descriptors_vlad.append(descrs)
                idImage.append(imagePath)
            else:
                print "ignore " + imagePath
    else:
        pool = multiprocessing.Pool(threads)

        data = [(f, visualDictionary, projection) for f in files]

        for descrs, imPath in tqdm(pool.imap_unordered(encode_one_image, data),
                                  desc="[{} CPUs] Calculating VLAD descriptors".format(threads), total=len(data)):
            if descrs is not None:
                descriptors_vlad.append(descrs)
                idImage.append(imPath)
            else:
                print "ignore " + imPath

        pool.close()

    #list to array
    descriptors_vlad = np.asarray(descriptors_vlad)
    return descriptors_vlad, idImage


def VLAD(X, visualDictionary):
    """
    Compute VLAD descriptors.

    :param X: descriptors of an image (M x D matrix)
    :param visualDictionary: precomputed visual dictionary
    :return:
    """
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in xrange(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    V = V/np.sqrt(np.dot(V,V))
    return V


def improvedVLAD(X,visualDictionary):
    """
    Implementation of an improved version of the VLAD descriptors.

    Reference: Delhumeau, J., Gosselin, P.H., Jegou, H. and Perez, P., 2013, October.
    Revisiting the VLAD image representation.
    In Proceedings of the 21st ACM international conference on Multimedia (pp. 653-656). ACM.

    :param X:
    :param visualDictionary:
    :return:
    """
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in xrange(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    V = V/np.sqrt(np.dot(V,V))

    return V


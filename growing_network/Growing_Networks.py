"""
.. moduleauthor::  Luiza Mici <miciluiza@gmail.com>

"""

import numpy as np
import cPickle as pickle
import copy

from .utils import pca
from .utils import squared_distances_sklearn_based


class Growing_Network:
    """
    Interface for Growing Network
    """

    def __init__(self):

        print "Using GWR of class:" ,self.__class__.__name__

        self.distance_function = squared_distances_sklearn_based

        self.last_dataset_index = 0
        self.clean_network = False
        self.epoch = 0
        self.iteration_index = 0
        self.edges = {}
        self.nodes_neighbor = {}

        self.nr_nodes = 0
        self.nodes_unique_key = 0
        self.nr_edges = 0
        self.max_age = 100

        self.nodes_keys = []  # = node_keys
        self.node_index_to_key = {}
        self.key_to_node_index = {}
        self.edges_to_remove = set([])
        self.node_to_remove = set([])

        # keep track of the eliminated/added nodes
        # during one learning iteration
        self.deleted_node_key_buffer = []
        self.added_node_key_buffer = []
        self.deleted_node_index_buffer = []

        self.quantization_error = []
        self.weights = None
        self.pca_weights = None
        self.squared_data = None
        self.track_qe = False
        self.name = ""

    def get_weights(self):
        """
        Returns weights associated with the network's nodes.

        :rtype: numpy array

        """
        return self.weights

    def get_pca_weights(self, pc=4):
        """
        Returns neurons weights with a dimensionality reduced through PCA

        :param pc: number of principal components
        :type pc: int

        """
        if self.weights.shape[1] == pc:
            self.pca_weights = self.weights

        if self.weights is not None and \
                (self.pca_weights is None or self.pca_weights.shape[1] != pc):
            three_dims_weights, _, _ = pca(data=self.weights, dims_rescaled_data=pc)
            self.pca_weights = three_dims_weights

        return self.pca_weights

    def initialise_network(self, **kwargs):
        """
        .. note::

            This is an interface function that does nothing. Each child class has to overwrite it!

        :param kwargs:

        :raises: Exception("Illegal State Exception!")

        """
        raise Exception("Illegal State Exception!")

    def create_node(self):
        """
        Adds a new node to the network. This method takes care of assigning the new node with
        a unique key and updating the two maps ``self.node_index_to_key`` and ``self.key_to_node_index``.
        It initializes with an empty list the node's neighbors and with an empty dictionary the node's edges.

        :return: new node key
        """
        node_key = self.nodes_unique_key
        self.nodes_unique_key += 1

        self.nodes_keys.append(node_key)

        self.nodes_neighbor[node_key] = []
        self.edges[node_key] = {}

        self.nr_nodes += 1

        node_idx = self.nodes_keys.index(node_key)
        self.node_index_to_key[node_idx] = node_key
        self.key_to_node_index[node_key] = node_idx

        self.added_node_key_buffer.append(node_key)

        return node_key

    def add_edge(self, key_node_i, key_node_j):
        """
        Adds and edge between two nodes. Edges have no direction!

        :param key_node_i: one node end of the edge
        :param key_node_j: second node end of the edge
        :type key_node_j: int
        :type key_node_i: int

        """
        if key_node_j in self.edges[key_node_i].keys() and key_node_i in self.edges[key_node_j].keys():
            # Edge exists - set age to 0;
            self.edges[key_node_i][key_node_j] = 0
            self.edges[key_node_j][key_node_i] = 0
        else:
            # else create edge and neighbor
            self.edges[key_node_i][key_node_j] = 0
            self.edges[key_node_j][key_node_i] = 0
            self.nr_edges += 1

        self.add_neighbor(key_node_i, key_node_j)

    def add_neighbor(self, key_node_i, key_node_j):
        """
        Make the two neurons *i* and *j* neighbor of each other (if they aren't already).

        :param key_node_i: first neuron whose neighbors will be updated
        :param key_node_j: second neuron whose neighbors will be updated
        :type key_node_j: int
        :type key_node_i: int

        """
        if key_node_j not in self.nodes_neighbor[key_node_i]:
            self.nodes_neighbor[key_node_i].append(key_node_j)
        if key_node_i not in self.nodes_neighbor[key_node_j]:
            self.nodes_neighbor[key_node_j].append(key_node_i)

    def find_bmus(self, **kwargs):
        """
        Find best matching units.

        .. note::

            This is an interface function that does nothing. Each child class has to overwrite it!

        :param kwargs:

        :raises: Exception("Illegal State Exception!")

        """
        raise Exception("Illegal State Exception!")

    def increment_age(self, key_node_i, key_node_j):
        """
        Increments the age of the edge i<-->j.
        Checks if the maximum age has been reached for the current edge, if so
        the edge is added to ``self.edges_to_remove`` list.

        :param key_node_i: one end neuron
        :param key_node_j: the second end neuron
        :type key_node_j: int
        :type key_node_i: int

        """
        self.edges[key_node_i][key_node_j] += 1
        self.edges[key_node_j][key_node_i] += 1
        if self.edges[key_node_i][key_node_j] >= self.max_age:
            self.edges_to_remove.add((key_node_i, key_node_j))

    def remove_old_connection(self):
        """
        Calls :func:`Growing_Network.remove_edge` on all edges within ``self.edges_to_remove`` list.
        When finished it empties the list.

        """
        for i_j in self.edges_to_remove:
            self.remove_edge(i_j[0], i_j[1])
        self.edges_to_remove = set([])

    def remove_edge(self, key_node_i, key_node_j):
        """
        Deletes the edge i<-->j and updates the nodes' neighbors accordingly.
        When edges are deleted, the nodes that were connected are candidates for ``self.node_to_remove``.
        This method checks all possible candidates.

        :param key_node_i: one end node
        :param key_node_j: the second end node
        :type key_node_j: int
        :type key_node_i: int

        """
        if key_node_j not in self.edges[key_node_i]:
            return

        self.edges[key_node_i].pop(key_node_j)
        self.edges[key_node_j].pop(key_node_i)

        # self.node_to_remove contains all node to remove when call remove_node
        if key_node_i in self.nodes_neighbor:
            if key_node_j in self.nodes_neighbor[key_node_i]:
                self.nodes_neighbor[key_node_i].remove(key_node_j)
            if not self.nodes_neighbor[key_node_i]:
                self.node_to_remove.add(key_node_i)

        if key_node_j in self.nodes_neighbor:
            if key_node_i in self.nodes_neighbor[key_node_j]:
                self.nodes_neighbor[key_node_j].remove(key_node_i)
            if not self.nodes_neighbor[key_node_j]:
                self.node_to_remove.add(key_node_j)

    def delete_nodes(self):
        """
        Deletes all nodes in ``self.node_to_remove`` list.
        The method controls also if in the meantime the node candidate was connected,
        in this case this node will not be removed.
        After deleting the unconnected nodes this method updates the ``self.node_index_to_key`` and
        ``self.key_to_node_index`` maps.

        :return: list of nodes indexes to remove
        :rtype: list

        """

        # first double check if node have edge in order to delete all edge from/to this node
        tmp = [n for n in self.node_to_remove]
        for key in tmp:
            if not self.nodes_neighbor[key] and not not self.edges[key]:
                # if node don't have neighbors but have edges
                edges_from_this_key = self.edges[key].keys()
                for e in edges_from_this_key:
                    self.remove_edge(key, e)

        index_node_to_remove = []

        for key in self.node_to_remove:
            if not not self.edges[key]:
                continue
            self.nodes_keys.remove(key)
            if key in self.nodes_neighbor:
                del self.nodes_neighbor[key]
            del self.edges[key]
            self.deleted_node_key_buffer.append(key)
            # print "Deleted node with key ", key
            self.nr_nodes -= 1
            __index = self.get_node_index(key)
            self.deleted_node_index_buffer.append(__index)
            index_node_to_remove.append(__index)

        if self.node_to_remove:
            self.key_to_node_index = {}
            self.node_index_to_key = {}
            for idx, key in enumerate(self.nodes_keys):
                self.key_to_node_index[key] = idx
                self.node_index_to_key[idx] = key

        self.node_to_remove = set([])

        return index_node_to_remove

    def get_node_key(self, idx):
        """
        The node key is unique. The index changes according to insertion and removing of neurons.
        The mapping index-->key is given by ``self.node_index_to_key``

        :param idx: index of the node
        :type idx: int

        :return: the key of the node with index *idx*
        """
        return self.node_index_to_key[idx]

    def get_node_index(self, key):
        """
        The node key is unique. The index changes according to insertion and removing of neurons.
        The mapping key-->index is given by ``self.key_to_node_index``

        :param key: key of the node
        :type key: int

        :return: the index of the node with key *key*
        """
        return self.key_to_node_index[key]

    def plot_qe(self):
        if not self.quantization_error or not self.track_qe:
            raise Exception("Quantization error not tracked, please call train with track_qe = True!")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.suptitle("Quantization Error over epochs.")
        plt.plot(self.quantization_error)
        plt.show()

    def initialize_squared_data(self, x):
        """
        A utility method that computes x*x (dot product).
        When performing batch training, this method optimizes the computation of the distance between
        a data sample and network weights.

        :param x: the data samples
        :type x: list
        """
        self.squared_data = {i: np.dot(x[i], x[i]) for i in xrange(len(x))}

    def train(self, x, y=None, max_epoch=50, track_qe=False, random_initialization=True, verbose=True):
        """
        Trains the network.

        - Checks if the weights have been already initialized (the case when this method is being called multiple times)
        - Initialize weights if necessary
        - Calls :func:`Growing_Network.initialize_squared_data`
        - Iterates for *max_epoch* nr of epochs
        - Calls :func:`Growing_Network.one_step_train` for the current data sample and associated label if available

        .. note::

            This is an interface function that does nothing. Each child class has to overwrite
            :func:`Growing_Network.one_step_train` in order for this method to be effective!

        :param x: the data samples
        :type x: ndarray
        :param y: the labels
        :type y: ndarray
        :param max_epoch: number of maximum epochs to run the training
        :type max_epoch: int
        :param track_qe: flag for tracking the quantization error of bmus during each learning iteration
        :type track_qe: bool
        :param random_initialization: flag for the random initialization of the weights
        :type random_initialization: bool
        :param verbose: flag for verbosity of this method
        :type verbose: bool

        :raises: Warning(" warning: weights already trained! if you continue they will be re-initialized!!!!!")

        """
        if verbose:
            print "Training..."

        if self.weights is not None and len(self.weights) > 2:
            raise Warning(" warning: weights already trained! if you continue they will be re-initialized!!!!!")

        if random_initialization:
            self.weights = np.asarray(np.random.random_sample((self.nr_nodes, np.array(x).shape[1])))
        else:
            self.weights = copy.deepcopy(np.asarray(x[:self.nr_nodes]))

        self.initialize_squared_data(x)
        data_length = len(x)
        self.track_qe = track_qe

        for epoch in xrange(max_epoch):

            dataset_indices = np.arange(0, data_length)

            # NOTE Uncomment this if you need to shuffle data
            # np.random.shuffle(dataset_indices)
            # print Warning('The data is being shuffled before learning!')

            self.last_dataset_index = dataset_indices[-1]
            self.epoch = epoch
            self.clean_network = True

            if verbose and (epoch % 10 == 0 or epoch == max_epoch - 1):
                print ("Epoch: " + str(epoch) + "/" + str(max_epoch - 1)), "->",
            for data_index, t in enumerate(dataset_indices):

                self.iteration_index = self.epoch * data_length + data_index

                dataset_sample = x[t]
                if y is not None and len(y) > 0:
                    label = y[t]
                else:
                    label = None

                if ~np.isfinite(dataset_sample[0]).all():
                    # skip nan or inf value
                    continue

                self.one_step_train(dataset_sample, label, t)
                self.clean_network = False

            if verbose and (epoch % 10 == 0 or epoch == max_epoch - 1):
                print self.nr_nodes, "neurons"

        if verbose:
            print "done."

    def train_and_labeling(self, x, y, max_epoch, track_qe=False, **kwargs):
        """
        .. note::

            This is an interface function that does nothing. Each child class has to overwrite it!

        :param x: the data samples
        :type x: list
        :param y: the labels
        :type y: list
        :param max_epoch: number of maximum epochs to run the training
        :type max_epoch: int
        :param track_qe: flag for tracking the quantization error of bmus during each learning iteration
        :type track_qe: bool

        :raises: Exception("Illegal State Exception!")

        """
        raise Exception("Illegal State Exception!")

    def one_step_train(self, dataset_sample, label, t):
        """
        Trains the network with the current data sample.

        .. note::

            This is an interface function that does nothing. Each child class has to overwrite it!

        :param dataset_sample: one data sample
        :param label: the associated label
        :param t: training iteration

        :raises: Exception("Illegal State Exception!")

        """
        raise Exception("Illegal State Exception!")

    def classify(self, data):
        """
        .. note::

            This is an interface function that does nothing. Each child class has to overwrite it!

        :param data: the data samples

        :raises: Exception("Illegal State Exception!")

        """
        raise Exception("Illegal State Exception!")

    @classmethod
    def load_network(cls, path=None):
        # Alternate constructor
        import cPickle as pickle
        if path is not None:
            filename = path
        else:
            filename = str(cls)
        with open(filename + ".p", 'rb') as f:
            network = pickle.load(f)
            if not str(network.__class__) == str(cls):
                raise Exception("The file" + filename + " is not saved instance of " + str(cls))
            return network

    @classmethod
    def init_from_file(cls, path=None):
        return cls.load_network(path)

    def save_network(self, path=None):
        """
        Saves the network in the provided filesystem path as a pickle file.

        :param path: the absolute filesystem path
        :type path: str
        """
        if path is not None:
            filename = path
        else:
            filename = str(self.__class__)
        with open(filename + ".p", 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if not set(other.__dict__.keys()) == set(self.__dict__.keys()):
                return False
            for key in self.__dict__.keys():
                if not isinstance(self.__dict__[key], np.ndarray):
                    if not self.__dict__[key] == other.__dict__[key]:
                        return False
                elif not np.equal(self.__dict__[key], other.__dict__[key]).all():
                    return False

            return True
        else:
            return False


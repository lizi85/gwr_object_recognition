"""
.. moduleauthor:: Luiza Mici <miciluiza@gmail.com>

"""

import math
import numpy as np

from Growing_Networks import Growing_Network
from .utils import argmin_and_second_argmin


class GWR(Growing_Network):
    """
    Implementation of the GWR algorithm
    """
    def __init__(self):
        """
        Overrides :func:`Growing_Networks.Growing_Network.__init__
        """
        Growing_Network.__init__(self)

        self.epsilon_b = None
        self.epsilon_n = None
        self.tau_b = None
        self.tau_n = None
        self.habituation_threshold = None
        self.insertion_threshold = None
        self.max_nodes = None
        self.activity = None
        self.habituation = None

        self.max_neighbor = 4

        self.dataset_index = 0
        self.labels = {}

        self.bmu1_key = 0
        self.bmu2_key = 1

    def initialise_network(self,
                           habituation_threshold=.1, insertion_threshold=.9,
                           max_nodes=900, epsilon_b=.1, tau_b=.3, epsilon_n=.01, tau_n=.1,
                           max_age=100, max_neighbor=4):
        """
        Sets the learning parameters and Initializes the network by creating the two initial nodes.

        :param habituation_threshold:
        :param insertion_threshold:
        :param max_nodes: maximum number of nodes allowed during learning
        :param epsilon_b: learning rate for the bmus
        :param tau_b: constant for the habituation decrease curve function for the bmus
        :param epsilon_n: learning rate for the bmus' neighbors
        :param tau_n: constant for the habituation decrease curve function for the bmus' neighbors
        :param max_age: maximum age for edges, when reached edges are removed
        :param max_neighbor: maximum number of neighbors for each neuron (affected by the weight updates)
        """

        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.tau_b = tau_b
        self.tau_n = tau_n

        self.habituation_threshold = habituation_threshold
        self.insertion_threshold = insertion_threshold
        self.max_nodes = max_nodes
        self.max_neighbor = max_neighbor
        self.max_age = max_age

        # initialize the network with 2 unlabeled nodes
        self.create_node()
        self.create_node()
        self.labels = {}

        self.activity = np.zeros(2)
        self.habituation = np.ones(2)

    def remove_neurons_without_neighbors(self):
        """
        Delete all neurons with no neighbors!

        """
        self.node_to_remove.update([key for key in self.nodes_neighbor.keys() if not self.nodes_neighbor[key]])
        if len(self.node_to_remove) > 0:
            indices_to_remove = self.delete_nodes()
            self.weights = np.delete(self.weights, indices_to_remove, 0)
            self.activity = np.delete(self.activity, indices_to_remove, 0)
            self.habituation = np.delete(self.habituation, indices_to_remove, 0)

    def add_neighbor(self, key_node_i, key_node_j):
        """
        Overrides :func:`Growing_Networks.Growing_Network.add_neighbor`.
        Make the two neurons *i* and *j* neighbor of each other iif the max nr of neighbors has not been exceeded.

        :param key_node_i: first neuron whose neighbors will be updated
        :param key_node_j: second neuron whose neighbors will be updated
        :type key_node_j: int
        :type key_node_i: int

        """
        if len(self.nodes_neighbor[key_node_i]) < self.max_neighbor and key_node_j not in self.nodes_neighbor[key_node_i]:
            self.nodes_neighbor[key_node_i].append(key_node_j)
        if len(self.nodes_neighbor[key_node_j]) < self.max_neighbor and key_node_i not in self.nodes_neighbor[key_node_j]:
            self.nodes_neighbor[key_node_j].append(key_node_i)

    def adapt_bmu_weights_and_his_neighbors(self, dataset_sample, bmu1_idx, bmu1_key):
        """
        - Updates weights and decreases habituation of the best matching unit and its neighbors
        - Ages all edges with an end to the bmu1

        :param dataset_sample: a data sample
        :type dataset_sample: numpy array
        :param bmu1_idx: index of the first best matching unit
        :param bmu1_key: key of the first best matching unit (key is passed as argument to \
        avoid access to ``self.node_index_to_key``)

        """
        self.weights[bmu1_idx] += self.habituation[bmu1_idx] * self.epsilon_b * (dataset_sample -
                                                                                 self.weights[bmu1_idx])
        self.habituation[bmu1_idx] += self.tau_b * 1.05 * (1 - self.habituation[bmu1_idx]) - self.tau_b

        for n_key in self.nodes_neighbor[bmu1_key]:
            n_index = self.get_node_index(n_key)
            self.weights[n_index] += self.habituation[n_index] * self.epsilon_n * (dataset_sample -
                                                                                   self.weights[n_index])
            self.habituation[n_index] += self.tau_n * 1.05 * (1 - self.habituation[n_index]) - self.tau_n

        # Step 8: Age the edges from this bmu
        map(lambda n_key: self.increment_age(bmu1_key, n_key), self.edges[bmu1_key])

    def add_node_and_update_edges(self, dataset_sample, bmu1_idx, bmu1_key, bmu2_key):
        """
        - Adds a new node to the network with weights ``(weights[bmu1_idx] + data_sample)/2`` .
        - Creates edges: new_node<-->bmu1 and new_node<-->bmu2, removes the edge bmu1<-->bmu2,

        :param dataset_sample: the data sample that has caused the adding of a neuron
        :param bmu1_idx: index of the first best matching unit
        :param bmu1_key: key of the first best matching unit (key is passed as argument to \
        avoid access to ``self.node_index_to_key`` )
        :param bmu2_key: key of the second best matching unit

        :return: new_node_created, new_node_key
        :rtype: bool, int
        """

        new_node_created = False
        new_node_key = None
        if self.nr_nodes < self.max_nodes:
            new_node_key = self.create_node()

            # add edge between new node and bmu1
            self.add_edge(new_node_key, bmu1_key)

            # add edge between new node and bmu2
            self.add_edge(new_node_key, bmu2_key)

            # remove edge between bmu1 and bmu2
            self.remove_edge(bmu1_key, bmu2_key)

            # update matrix
            self.activity = np.hstack((self.activity, np.zeros(1)))
            self.habituation = np.hstack((self.habituation, np.ones(1)))

            new_weights = (self.weights[bmu1_idx] + dataset_sample) / 2.
            self.weights = np.vstack((self.weights, new_weights))

            new_node_created = True

        return new_node_created, new_node_key

    def find_bmus(self, data_sample):
        """
        Overrides :func:`Growing_Networks.Growing_Network.find_bmus`.
        Finds the first and the second best matching unit for the given data sample.

        :param data_sample: the data sample
        :type data_sample: numpy array

        :return first bmu, second bmu, smallest distance

        """
        if self.dataset_index in self.squared_data:
            square_data = self.squared_data[self.dataset_index]
        else:
            square_data = None
        return argmin_and_second_argmin(
            self.distance_function(x=data_sample, y=self.weights, x_norm_squared=square_data))

    def compute_activity(self, node_idx, distance):
        """
        Computes activity of the node as: ``math.exp(-(distance ** 2))``

        :param node_idx: index of the node
        :type node_idx: int
        :param distance: distance with data sample
        :type distance: float
        """
        self.activity[node_idx] = math.exp(-(distance ** 2))

    def train(self, x, y=None, max_epoch=50, track_qe=False, random_initialization=True, verbose=True):
        """
        Calls :func:`Growing_Networks.Growing_Network.train` and when training is finished removes old connections
        and neurons without connections.

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

        """
        Growing_Network.train(self, x, y, max_epoch, track_qe, random_initialization, verbose)
        # final clean
        self.remove_old_connection()
        self.remove_neurons_without_neighbors()

    def one_step_train(self, dataset_sample, label, t):
        """
        Implements :func:`Growing_Networks.Growing_Network.one_step_train`.

        - Finds first and second best matching units
        - Adds an edge between them
        - Computes activity of the first best matching unit bmu1
        - If habituation[bmu1] is lower than the habituation threshold and activity[bmu1] is lower than the \
        insertion_threshold adds a neuron, otherwise trains the bmu1 and its neighbors


        :param dataset_sample: a 1D vector
        :type dataset_sample: numpy array
        :param label: label of the current data sample
        :param t: training iteration
        """
        self.dataset_index = t

        # Find the best and second-best matching neurons
        bmu1_idx, bmu2_idx, distance = self.find_bmus(dataset_sample)
        distance = math.sqrt(abs(distance))

        if self.track_qe:
            self.quantization_error.append(distance)

        bmu1_key = self.get_node_key(bmu1_idx)
        bmu2_key = self.get_node_key(bmu2_idx)

        self.bmu1_key = bmu1_key
        self.bmu2_key = bmu2_key

        # add edge between best matching units
        self.add_edge(bmu1_key, bmu2_key)

        # calculate the activity of the best matching unit
        self.compute_activity(bmu1_idx, distance)

        if self.habituation[bmu1_idx] < self.habituation_threshold \
                and self.activity[bmu1_idx] < self.insertion_threshold\
                and self.max_nodes > self.nr_nodes:

            # Add new node and update edges
            _, new_key_node = self.add_node_and_update_edges(dataset_sample, bmu1_idx, bmu1_key, bmu2_key)

            # set label if label strategy = online
            self.set_label_for_node(new_key_node, label)

        else:
            # If no new nodes, adapt weights and habituate for bmu1 and his neighbors, then age edges
            self.adapt_bmu_weights_and_his_neighbors(dataset_sample, bmu1_idx, bmu1_key)

            # set label if label strategy = online
            self.set_label_for_node(bmu1_key, label)

            # Remove old connections
            self.remove_old_connection()

        if self.squared_data is None or t == self.last_dataset_index or self.clean_network or t % 1 == 0:

            # remove neurons without connections
            self.remove_neurons_without_neighbors()

        self.dataset_index = -1

    def set_label_for_node(self, key, label):

        if label is not None:

            if key not in self.labels:
                self.labels[key] = [label]
            else:
                self.labels[key].append(label)

    def get_labels(self):
        """
        Returns neurons' labels. If a neuron has no label, -1 is returned.

        :rtype: dict
        """

        import scipy.stats as scp
        _labels = {key: -1 for key in self.nodes_keys}

        for key in _labels.keys():

            try:
                values = self.labels[key]
            except:
                values = []
                print "labels for node key " + str(key) + " not found!"

            if len(values) > 0:
                _labels[key] = scp.mode(values)[0][0]

        return _labels

    def classify(self, x):
        """
        Finds distances between each data sample in x with the nodes in the network and
        returns best matching units as well as the labels assigned to each bmu.

        :param x: the data samples
        :return: bmus, labels, distances
        """
        output_labels = []
        bmus = []
        distances = []
        labels = self.get_labels()

        for i in xrange(len(x)):

            data_sample = x[i]
            bmu_idx, _, distance = self.find_bmus(data_sample)
            bmu_key = self.get_node_key(bmu_idx)
            bmus.append(bmu_idx)
            distances.append(distance)
            output_labels.append(int(labels[bmu_key]))

        return bmus, output_labels, distances

    # noinspection PyUnresolvedReferences
    def plot_graph(self, colors, node_labels, classes, weights=None, with_cmap=False):

        """
        Plots a 3-dimensional graph, where nodes have coordinates of weights and color and label specified in the args.
        Connection between nodes is defined in self.edges. Colors of the nodes can be class related or they represent
        the activity of the network corresponding to a specific input vector.

        :param with_cmap:
        :param colors: numpy array of colors associated to each node
        :param node_labels: numpy array of labels associated to each node
        :param weights: if None the matrix self.weights is used
        :param classes:
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if weights is None:
            weights = self.weights

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        for j in xrange(len(classes)):

            class_indexes = (node_labels == classes[j]).reshape(-1)
            _weights_for_class = weights[class_indexes]
            _colors_for_class = colors[class_indexes]
            if with_cmap:
                ax.scatter(_weights_for_class[:, 0], _weights_for_class[:, 1], _weights_for_class[:, 2],
                           s=300, c=_colors_for_class[:, 0], label=str(classes[j]), cmap='Oranges')
            else:
                ax.scatter(_weights_for_class[:, 0], _weights_for_class[:, 1], _weights_for_class[:, 2],
                           s=300, c=_colors_for_class, label=str(classes[j]))

        ax.legend()

        for key, values in self.edges.iteritems():
            for value in values:
                node1 = self.get_node_index(key)
                node2 = self.get_node_index(value)
                ax.plot([weights[node1][0], weights[node2][0]],
                        [weights[node1][1], weights[node2][1]],
                        [weights[node1][2], weights[node2][2]], linewidth=1.0, color='gray')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        plt.show()

    def plot_graph_2d(self, fig=None):
        import matplotlib.pyplot as plt

        # weights = self.get_pca_weights(pc=2)
        weights = self.weights

        if fig is None:
            fig = plt.figure()

        ax = fig.gca()
        ax.scatter(weights[:, 0], weights[:, 1], color='black')

        for key, values in self.edges.iteritems():
            for value in values:
                node1 = self.get_node_index(key)
                node2 = self.get_node_index(value)
                ax.plot([weights[node1][0], weights[node2][0]],
                        [weights[node1][1], weights[node2][1]], linewidth=1.0, color='gray')

        ax.set_xlabel('X1', fontsize=20)
        ax.set_ylabel('X2', fontsize=20)

        plt.show()

    def plot_graph_2d_with_labels(self, colors, node_labels, classes, weights=None, with_cmap=False, target_names=None):

        """
        Plots a 2-dimensional graph, where nodes have coordinates of weights and color and label specified in the args.
        Connection between nodes is defined in self.edges. Colors of the nodes can be class related or they represent
        the activity of the network corresponding to a specific input vector.

        :param with_cmap:
        :param colors: numpy array of colors associated to each node
        :param node_labels: numpy array of labels associated to each node
        :param weights: if None the matrix self.weights is used
        :param classes:
        """

        import matplotlib.pyplot as plt

        if weights is None:
            weights = self.weights

        font = {'family': 'Times New Roman', 'size': 18}

        plt.rc('font', **font)

        fig = plt.figure()
        ax = fig.gca()

        if target_names is None:
            target_names = ['biscuitsbox', 'can', 'mug', 'phone']
        for j in xrange(len(classes)):

            # class_indexes = (node_labels == classes[j]).reshape(-1)
            class_indexes = [i for i, x in enumerate(node_labels) if x == classes[j]]
            _weights_for_class = weights[class_indexes]
            _colors_for_class = colors[class_indexes]
            if with_cmap:
                ax.scatter(_weights_for_class[:, 0], _weights_for_class[:, 1],
                           s=50, c=_colors_for_class[:, 0], label=str(classes[j]), cmap='Oranges')
            else:
                ax.scatter(_weights_for_class[:, 0], _weights_for_class[:, 1],
                           s=50, c=_colors_for_class, label=str(target_names[classes[j]]))

        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, borderaxespad=0.)

        ax.set_xlabel('PC1', fontsize=20)
        ax.set_ylabel('PC2', fontsize=20)

        plt.show()

    def plot_clusters(self):

        import matplotlib.cm as cm

        labels = self.get_labels()
        for node, label in labels.iteritems():
            if np.array(label).ndim == 1:
                labels[node] = np.argmax(label)

        unique_labels = np.unique(labels.values())
        if len(unique_labels) < max(unique_labels) + 1:
            classes = np.arange(0, max(unique_labels)+1)
        else:
            classes = unique_labels

        random_colors = cm.rainbow(np.linspace(0, 1, max(classes)+2))

        # sort labels to match indexes of self.weights matrix
        sorted_labels = np.empty((self.weights.shape[0], 1))
        colors = np.empty((sorted_labels.shape[0], 4))
        for i in xrange(len(self.weights)):
            _label = labels[self.get_node_key(i)]
            sorted_labels[i] = _label
            colors[i] = random_colors[np.where(classes == _label)]

        self.plot_graph_2d_with_labels(colors, sorted_labels, classes, self.get_pca_weights(pc=2), target_names=classes)


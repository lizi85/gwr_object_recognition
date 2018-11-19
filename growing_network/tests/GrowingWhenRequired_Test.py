import unittest

from sklearn import datasets
from sklearn.metrics import confusion_matrix

from growing_network.GrowingWhenRequired import *

np.random.seed(42)


class GWR_test(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.gwr = GWR()
        self.gwr.initialise_network()

    def test_adapt_bmu_weights_and_his_neighbors(self):
        # given
        dataset = np.random.random_sample(1)

        a = self.gwr.create_node()
        b = self.gwr.create_node()
        c = self.gwr.create_node()

        a_idx = self.gwr.get_node_index(a)
        b_idx = self.gwr.get_node_index(b)
        c_idx = self.gwr.get_node_index(c)

        self.gwr.add_edge(a, b)
        self.gwr.add_edge(a, c)

        self.gwr.weights = np.array([[0], [0], [1.], [1.], [1.]])
        self.gwr.habituation = np.array([0, 0, 1., 1., 1.])

        # when
        self.gwr.adapt_bmu_weights_and_his_neighbors(dataset, a_idx, a)

        # then
        assert self.gwr.weights[a_idx][0] < .937454011886
        assert self.gwr.weights[b_idx][0] < .993745401189
        assert self.gwr.weights[b_idx][0] == self.gwr.weights[c_idx][0]

        assert self.gwr.habituation[a_idx] == .7
        assert self.gwr.habituation[b_idx] == .9
        assert self.gwr.habituation[b_idx] == self.gwr.habituation[c_idx]

        assert self.gwr.edges[a_idx][b_idx] == 1
        assert self.gwr.edges[a_idx][c_idx] == 1

        assert self.gwr.edges[b_idx][a_idx] == 1
        assert self.gwr.edges[c_idx][a_idx] == 1

    def test_add_node_and_update_edges(self):
        # given
        dataset = np.random.random_sample(1)

        a = self.gwr.create_node()
        b = self.gwr.create_node()
        c = self.gwr.create_node()

        a_idx = self.gwr.get_node_index(a)

        self.gwr.weights = np.array([[0], [0], [1.], [1.], [1.]])
        self.gwr.habituation = np.array([0, 0, 1., 1., 1.])
        self.gwr.activity = np.array([0, 0, 1., 1., 1.])

        # when
        r, k = self.gwr.add_node_and_update_edges(dataset, a_idx, a, b)

        # then
        assert r == True
        self.assertIsNotNone(k)
        assert self.gwr.nr_nodes == 6
        assert self.gwr.edges[a][5] == 0
        assert self.gwr.edges[5][a] == 0

        assert self.gwr.nr_nodes == len(self.gwr.activity)
        assert self.gwr.nr_nodes == len(self.gwr.habituation)
        assert self.gwr.nr_nodes == len(self.gwr.weights)

    def test_train_should_not_fail(self):
        from sklearn import datasets
        iris = datasets.load_iris()
        dataset = iris.data
        labels = iris.target
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(n_values=3)
        enc.fit(labels.reshape((len(labels), 1)))

        self.gwr = GWR()
        self.gwr.initialise_network(habituation_threshold=1.1, insertion_threshold=1,
                                    max_nodes=500, epsilon_b=.1, tau_b=.3, epsilon_n=.01, tau_n=.1,
                                    max_age=100)

        self.gwr.train(x=dataset, max_epoch=50, track_qe=True)
        assert self.gwr.quantization_error[-1] < .04

    def test_train_iris(self):
        # import some data to play with
        iris = datasets.load_iris()
        dataset = iris.data
        # dataset = scale_data(iris.data)
        labels = iris.target
        my_gwr = GWR()
        my_gwr.initialise_network(habituation_threshold=.1, insertion_threshold=.99)

        my_gwr.train(x=dataset, y=labels, max_epoch=100)
        # print self.my_gwr.nodes
        _, best_matching_labels, _ = my_gwr.classify(x=dataset)

        from sklearn.metrics import classification_report
        y_true = labels
        y_pred = best_matching_labels
        target_names = ['setosa', 'versicolor', 'virginica']
        print(classification_report(y_true, y_pred, target_names=target_names))
        print '*******************************'
        print(confusion_matrix(y_true, y_pred))


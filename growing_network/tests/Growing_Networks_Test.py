import unittest

from growing_network.Growing_Networks import *


class Growing_Network_test(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.grow_network = Growing_Network()

    def test_create_node(self):
        i = self.grow_network.create_node()

        assert self.grow_network.nr_nodes == 1
        assert i == 0
        assert self.grow_network.nodes_counter[i] < 0.01
        assert self.grow_network.edges[i] == {}
        assert self.grow_network.nodes_neighbor[i] == []
        assert i in self.grow_network.nodes_keys
        assert i in self.grow_network.key_to_node_index
        assert self.grow_network.get_node_index(i) in self.grow_network.node_index_to_key

        i = self.grow_network.create_node()

        assert self.grow_network.nr_nodes == 2
        assert i == 1
        assert self.grow_network.nodes_counter[i] < 0.01
        assert self.grow_network.edges[i] == {}
        assert self.grow_network.nodes_neighbor[i] == []
        assert i in self.grow_network.nodes_keys
        assert i in self.grow_network.key_to_node_index
        assert self.grow_network.get_node_index(i) in self.grow_network.node_index_to_key

    def test_add_edge(self):
        # given
        # self.mgng.initialise_network(.5, .5, .5, .5, .5, .5, .5, .5)

        self.grow_network.create_node()
        self.grow_network.create_node()

        self.grow_network.create_node()
        self.grow_network.create_node()

        # when new edge
        self.grow_network.add_edge(0, 1)
        self.grow_network.add_edge(0, 2)
        self.grow_network.add_edge(0, 3)

        assert self.grow_network.nr_edges == 3
        assert 1 in self.grow_network.edges[0]
        assert 0 in self.grow_network.edges[1]

        assert 1 in self.grow_network.nodes_neighbor[0]
        assert 0 in self.grow_network.nodes_neighbor[1]

        # assert age is 0
        assert self.grow_network.edges[0][1] == 0
        assert self.grow_network.edges[1][0] == 0

        # when existing edge
        self.grow_network.edges[0][2] = 20
        self.grow_network.edges[2][0] = 20
        self.grow_network.add_edge(0, 2)

        assert self.grow_network.edges[0][2] == 0
        assert self.grow_network.edges[2][0] == 0

    def test_remove_old_connection(self):
        # given
        self.grow_network.max_age = .5

        self.grow_network.create_node()
        self.grow_network.create_node()

        i = self.grow_network.create_node()
        self.grow_network.add_edge(1, i)
        i = self.grow_network.create_node()
        self.grow_network.add_edge(1, i)

        self.grow_network.add_edge(1, 2)
        self.grow_network.increment_age(1, 2)
        self.grow_network.increment_age(1, 2)

        # when
        self.grow_network.remove_old_connection()

        # then
        assert 1 not in self.grow_network.nodes_neighbor[2]
        assert 2 not in self.grow_network.nodes_neighbor[1]

        assert 1 not in self.grow_network.edges[2]
        assert 2 not in self.grow_network.edges[1]

    def test_delete_node(self):
        # given
        a = self.grow_network.create_node()
        b = self.grow_network.create_node()
        c = self.grow_network.create_node()
        self.grow_network.add_edge(a, b)
        self.grow_network.add_edge(b, c)

        # when
        self.grow_network.remove_edge(a, b)

        # then
        assert a in self.grow_network.node_to_remove
        assert len(self.grow_network.node_to_remove) == 1

        # when
        self.grow_network.delete_nodes()

        # then
        assert self.grow_network.nr_nodes == 2
        assert a not in self.grow_network.nodes_counter
        assert a not in self.grow_network.nodes_neighbor
        assert a not in self.grow_network.edges
        assert a not in self.grow_network.nodes_keys
        assert a not in self.grow_network.key_to_node_index

    def test_delete_node_idx(self):
        # given
        a = self.grow_network.create_node()
        b = self.grow_network.create_node()
        c = self.grow_network.create_node()
        self.grow_network.add_edge(a, b)
        self.grow_network.add_edge(b, c)

        self.grow_network.remove_edge(c, b)

        keys = self.grow_network.node_to_remove
        key_node_to_remove = self.grow_network.get_node_index(next(iter(keys)))

        # when
        indices = self.grow_network.delete_nodes()

        # then
        assert key_node_to_remove == indices[0]

    def test_get_node_key(self):
        # given
        a = self.grow_network.create_node()
        b = self.grow_network.create_node()
        c = self.grow_network.create_node()

        self.grow_network.add_edge(a, c)
        self.grow_network.add_edge(a, b)

        a_idx = self.grow_network.nodes_keys.index(a)
        b_idx = self.grow_network.nodes_keys.index(b)
        c_idx = self.grow_network.nodes_keys.index(c)

        # then
        self.assertEqual(a_idx, 0)
        self.assertEqual(b_idx, 1)
        self.assertEqual(c_idx, 2)

        self.grow_network.remove_edge(a, b)
        self.grow_network.delete_nodes()

        # when
        a__idx = self.grow_network.get_node_index(a)
        c__idx = self.grow_network.get_node_index(c)

        # then
        assert c__idx != c_idx
        assert a__idx == a_idx

        assert c == self.grow_network.get_node_key(c__idx)
        assert a == self.grow_network.get_node_key(a__idx)
        assert b not in self.grow_network.key_to_node_index

        assert a_idx == self.grow_network.key_to_node_index[a]
        assert c_idx != self.grow_network.key_to_node_index[c]

        assert self.grow_network.key_to_node_index[c] == self.grow_network.nodes_keys.index(c)
        assert self.grow_network.key_to_node_index[a] == self.grow_network.nodes_keys.index(a)

        assert a == self.grow_network.node_index_to_key[self.grow_network.key_to_node_index[a]]
        assert c == self.grow_network.node_index_to_key[self.grow_network.key_to_node_index[c]]

    def test_should_save_load_from_file(self):
        # given
        grow_network = Growing_Network()

        grow_network.create_node()
        grow_network.create_node()
        grow_network.create_node()
        grow_network.add_edge(0, 1)

        grow_network.weights = np.random.random_sample(50)

        # when
        grow_network.save_network("/tmp/grow")
        loaded = Growing_Network.load_network("/tmp/grow")

        # then
        self.assertEqual(grow_network, loaded)

    def test_should_calculate_PCA_for_weights(self):
        # given
        self.grow_network.weights = np.random.random_sample((50, 50))

        # when
        weightsPCA = self.grow_network.get_pca_weights(pc=3)

        # then
        self.assertEqual(weightsPCA.shape, (50L, 3L))

        # when
        weightsPCA = self.grow_network.get_pca_weights(pc=4)

        # then
        self.assertEqual(weightsPCA.shape, (50L, 4L))

        # when
        weightsPCA = self.grow_network.get_pca_weights(pc=5)

        # then
        self.assertEqual(weightsPCA.shape, (50L, 5L))

    # ignore test
    def ___test_check_PCA(self):
        from sklearn.decomposition import PCA

        # given
        self.grow_network.weights = np.random.random_sample((50, 50))

        s = PCA(n_components=4, copy=False).fit_transform(self.grow_network.weights)

        # when
        weightsPCA = self.grow_network.get_pca_weights(pc=4)

        #TODO: check
        #then
        self.assertTrue( np.allclose( weightsPCA, s, rtol=1.e-5) )



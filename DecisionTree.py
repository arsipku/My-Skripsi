"""
DecisionTree.py
Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/
Self-implementation of a decision tree.
Package requirements:
numpy
pandas
"""

# Imports
import numpy as np
import pandas as pd
from collections import Counter


class _DecisionTree(object):
    """
    Abstract decision tree class.  Will use a linked-list-esque implementation where node is a "Node" class,
    and the node holds a cutoff, and then reference to another node.  Nodes can also hold a terminating value.
    Abstraction is made using the private (_) to not be able to implement outside.
    Parameters:
        self._type: classification or regression
        self._split_func: function to split on
        self._leaf_terminate: The leaf terminating criteria (how many points to left in the data to create
                             a leaf node). Defaults to 1 or None (if termination criteria = 'leaf'.
        self._pure_terminate: True/False pending if the termination criteria is pure for classification trees
        self._split_criteria: np.min/np.max depending on regression/classification
        self._arg_split_criteria: np.argmin/argmax depending on regression/classification
        self._prune: True/False if we want to prune or not
        self._node_list: A list of 2D nodes, where each section of the outer list is a level of the
                          tree, and then the lists within a level are the inidividual nodes at that level
        self._ncols: The number of features within the given dataset.
    Methods:
        Public
        ------
        Initialization: Initializes the class
        fit: Takes an inputted dataset and creates the decision tree.
        predict: Takes a new dataset, and runs the algorithm to perform a prediction.
        get_tree: Returns the tree with leaf nodes (self._node_list)
        handle_data: Handle/clean data
        Private
        -------
        Node Class: A node class (see node class for explanation)
        __recursive_fit: Fits a tree recursively by calculating a column to split on
        __create_node: Create a new node
        __split_data: Split the data between branches
        _create_new_nodes: Create a set of new nodes by splitting
        _find_split: Find the optimal splitting point
        __terminate_fit: Checks at a stage whether each leaf satisfies the terminating criteria
        _recursive_predict: Does the recursive predictions at each point
        _prune_tree: Abstract method not defined until later
    """
    def __init__(self, tree_type, split_func, leaf_terminate, pure_terminate, prune, split_criteria):
        """
        Initialize all parameters
        :param tree_type: classification vs. regression
        :param split_func: the function to tabulate error
        :param leaf_terminate: If a number, how many samples must be in a leaf to terminate
        :param pure_terminate: True/False, saying if one should terminate in classification if all classes are equal
        :param prune: Whether to use pessimistic pruning
        :param split_criteria: (np.min/np.max, np.argmin/np.argmax)
        """
        # Initialize all parameters
        self._type = tree_type
        self._split_func = split_func
        self._leaf_terminate = leaf_terminate
        self._pure_terminate = pure_terminate
        self._split_criteria = split_criteria[0]
        self._arg_split_criteria = split_criteria[1]
        self._prune = prune

        # Initialize the tree parameters
        self._ncols = 0
        self._node_list = []

    def fit(self, x_data, y_data):
        """
        Fit (train) the decision tree using an inputted dataset.
        :param x_data: The dataset to train the decision tree with.
        :param y_data: The result vector we are regressing on.
        """
        # Adjust the data
        x_data, y_data = self.handle_data(x_data, y_data)

        # Get number of columns
        self._ncols = x_data.shape[1]

        # Make initial node with all data
        self._node_list.append([self._create_node(x_data, y_data)])
        
        # Recursive fit
        self.__recursive_fit([0, 0])

        # Prune if necessary
        if self._prune:
            self._prune_tree([0, 0])

    @staticmethod
    def handle_data(x_data=None, y_data=None):
        """
        Handle the x/y data to make sure they are in arrays and have no-null values.
        :param x_data: The dataset to train the decision tree with.
        :param y_data: The result vector we are regressing on.
        """
        # Make into a numpy array if dataframe
        if (x_data is not None) and (type(x_data) != np.ndarray):
            if type(x_data) == pd.DataFrame:
                x_data = x_data.values
            elif type(x_data) == np.matrix:
                x_data = np.asarray(x_data)
            else:
                raise ValueError(
                    'X data input is not in correct format. X data must be 2-dimensional, and '
                    'X data can be a numpy array, matrix, pd.DataFrame.'
                )
        elif x_data.ndim == 1:
            x_data = x_data.reshape((1, x_data.shape[0]))

        if (y_data is not None) and (type(y_data) != np.ndarray):
            if type(y_data) == pd.Series:
                y_data = y_data.values
            elif type(y_data) == np.matrix:
                y_data = np.asarray(y_data).flatten()
            else:
                raise ValueError(
                    'Y data input is not in correct format. Y data must be one-dimensional, and '
                    'Y data can be a numpy array, matrix, pd.Series.'
                )

        return x_data, y_data

    def __recursive_fit(self, curr_idx):
        """
        Recursively fit nodes while not satisfying the terminating criteria.
        :param curr_idx: The current 2-d index the function is calling to
        """
        level, n = curr_idx[0], curr_idx[1]
        if not self.__terminate_fit(curr_idx):
            split_col, val = self._find_split(
                curr_idx,
                self._split_criteria,
                self._arg_split_criteria,
                self._split_func
            )
            self._node_list[level][n].set_split(split_col, val)
            # Create new nodes
            lower_idx, upper_idx = self._create_new_nodes(level, n)
            # Call the function if necessary
            if lower_idx[1] is not None:
                self.__recursive_fit(lower_idx)
            if upper_idx[1] is not None:
                self.__recursive_fit(upper_idx)

    def _create_node(self, x_data, y_data):
        """
        Creates new node and determines if it is a leaf node.
        :param x_data: The x data to create the node
        :param y_data: The prediction data to create the node
        :return: The new node object
        """
        # Return if leaf
        if self._pure_terminate:
            # Check y_data holds one unique value
            if len(np.unique(y_data)) > 1 and x_data.shape[0] > 1:
                return self._Node(False, x_data, y_data)
        else:
            # Check leaf size
            if x_data.shape[0] > self._leaf_terminate:
                return self._Node(False, x_data, y_data)

        # Return if branching node
        if self._type == 'classification':
            return self._Node(True, x_data, y_data, 'classification')
        else:
            return self._Node(True, x_data, y_data, 'regression')

    def _split_data(self, level, n, idx, split_val):
        """
        Split the data based upon a value.
        :param level: the level
        :param n: the node index in the level
        :param idx: the index to split on
        :param split_val: the split value
        :return: the split
        """
        x_data = self._node_list[level][n].get_x_data()
        y_data = self._node_list[level][n].get_y_data()

        lower_x_data = x_data[x_data[:, idx] < split_val]
        lower_y_data = y_data[x_data[:, idx] < split_val]
        upper_x_data = x_data[x_data[:, idx] >= split_val]
        upper_y_data = y_data[x_data[:, idx] >= split_val]

        return lower_x_data, lower_y_data, upper_x_data, upper_y_data

    def _create_new_nodes(self, level, n):
        """
        Create the next level of nodes. Splits the data based upon the specified axis, and
        creates the new level of nodes by splitting the data.
        :param level: The level value to create the new nodes on
        :param n: The index in the level we are on
        :return: the upper and lower tuples for the new nodes created
        """
        if (level + 1) == len(self._node_list):
            self._node_list.append([])

        split_val = self._node_list[level][n].get_split()
        idx = self._node_list[level][n].get_col()

        # Split data
        lower_x_data, lower_y_data, upper_x_data, upper_y_data = self._split_data(level, n, idx, split_val)

        # Now check if all the same in lower/upper
        # Do not change y_data to average over all values
        if (lower_x_data.shape[0] > 1) and ((lower_x_data - lower_x_data[0, :]) == 0).all():
            lower_x_data = lower_x_data[[0], :]
        if (upper_x_data.shape[0] > 1) and ((upper_x_data - upper_x_data[0, :]) == 0).all():
            upper_x_data = upper_x_data[[0], :]
        # Make lower node if one can
        if lower_x_data.shape[0] > 0:
            lower_curr_index = len(self._node_list[level + 1])
            self._node_list[level + 1].append(self._create_node(lower_x_data, lower_y_data))
            self._node_list[level][n].set_lower_split_index(lower_curr_index)
        else:
            lower_curr_index = None
        # Make upper node
        if upper_x_data.shape[0] > 0:
            upper_curr_index = len(self._node_list[level + 1])
            self._node_list[level + 1].append(self._create_node(upper_x_data, upper_y_data))
            self._node_list[level][n].set_upper_split_index(upper_curr_index)
        else:
            upper_curr_index = None

        return [level + 1, lower_curr_index], [level + 1, upper_curr_index]

    def _find_split(self, curr_idx, decision_func, arg_func, criteria_func):
        """
        Find split using the given criteria
        :param curr_idx: The current 2-d index the function is calling to
        :param decision_func: np.min/np.max depending if rss vs. gini
        :param arg_func: np.argmin/np.argmax depending if rss vs. gini
        :param criteria_func: either self.__rss__, self.__gini_impurity_gain__, or self.__gain_ratio__
        :return: the split column/value
        """
        level, n = curr_idx[0], curr_idx[1]
        x_data = self._node_list[level][n].get_x_data()
        col_min = []
        col_val = []
        for i in range(self._ncols):
            temp_desc = []
            temp_val = []
            temp_list = list(np.unique(x_data[:, i]))
            temp_list.sort()
            for j in range(len(temp_list) - 1):
                m = np.mean([temp_list[j], temp_list[j + 1]])
                temp_val.append(m)
                temp_desc.append(criteria_func(curr_idx[0], curr_idx[1], i, m))
            # Checks
            if len(temp_desc) == 0:
                if decision_func == np.min:
                    temp_desc.append(1e10)
                else:
                    temp_desc.append(-1e10)
                temp_val.append(0)
            col_min.append(decision_func(temp_desc))
            col_val.append(temp_val[arg_func(temp_desc)])

        return arg_func(col_min), col_val[arg_func(col_min)]

    def __terminate_fit(self, curr_idx):
        """
        Decide if fit is terminated.
        :param: The current 2D idx
        :return: True if terminated, False if not
        """
        if self._node_list[curr_idx[0]][curr_idx[1]].is_leaf():
            return True
        return False

    def _recursive_predict(self, args):
        """
        Follow the tree to get the correct prediction.
        :param args: The data we are predicting on, The current node we are looking at
        :return: The prediction
        """
        # Extract data
        x_data, curr_idx = args
        # Check if leaf
        if self._node_list[curr_idx[0]][curr_idx[1]].is_leaf():
            return self._node_list[curr_idx[0]][curr_idx[1]].get_prediction()
        else:
            # Figure out next leaf to look at
            idx = self._node_list[curr_idx[0]][curr_idx[1]].get_col()
            split = self._node_list[curr_idx[0]][curr_idx[1]].get_split()
            if x_data[idx] < split:
                new_idx = [curr_idx[0] + 1, self._node_list[curr_idx[0]][curr_idx[1]].get_lower_split()]
            else:
                new_idx = [curr_idx[0] + 1, self._node_list[curr_idx[0]][curr_idx[1]].get_upper_split()]
            return self._recursive_predict((x_data, new_idx))

    def _prune_tree(self, curr_idx=None):
        """
        Static method to be overwritten if a classification tree.
        """
        pass

    def predict(self, x_data):
        """
        Predict a class using the dataset given.
        :param x_data: The dataset to predict
        :return: A vector of predictions for each row in X.
        """
        # Clean data
        x_data, _ = self.handle_data(x_data)

        # Iteratively go through data
        input_list = [(x_data[i, :], [0, 0]) for i in range(x_data.shape[0])]

        return np.fromiter(map(self._recursive_predict, input_list), dtype=np.float)

    def get_tree(self):
        """
        Get the underlying tree object.
        :return: The tree (self._node_list())
        """
        return self._node_list

    class _Node(object):
        """
        Internal node class.  Used to hold splitting values, or termination criteria.
        All parameters are private since we do not any editing to take place after we setup the node.
        Parameters:
            self.__leaf: True if leaf node/False if not
            self.__x_data: The array of input data points for that node
            self.__y_data: The array of target data points for that node
            self.__prediction: The prediction, if it is a leaf
            self.__lower_split: If a non-leaf node, the reference to the place in the next level list for
                                  the coming node
            self.__upper_split: If a non-leaf node, the reference to the place in the next level list
                                  for the coming node
            self.__col: The column one is splitting on
            self.__split: The value to split on
        Methods:
        Public
        ------
        Initialization: Initialize a new node, and determine if it is a leaf
        is_leaf: True/false statement returning if this is a leaf
        prune: Removes the split values for a node and makes the node a leaf
        set_split: Set the split amount to branch the tree
        set_lower_split_index: Set the index to the node in the next level < split value
        set_upper_split_index: Set the index to the node in the next level > split value
        get_x_data: Get the x data specific to this node
        get_y_data: Get the y data specific to this node
        get_prediction: Get the y value for this node
        get_col: Get the column this node splits on
        get_split: Get the value this node splits on
        get_lower_split: Get the lower split index value
        get_upper_split: Get the upper split index
        """

        def __init__(self, leaf, x_data, y_data, leaf_type=None):
            """
            Initialize the node.
            :param leaf: The true/false value (defaults to false) to say if a node is a leaf
            :param x_data: The data to be placed into the node
            :param y_data: The y_data to be averaged over if a leaf node
            :param leaf_type: Either classification or regression
            """
            # Set leaf value
            self.__leaf = leaf
            self.__x_data = x_data
            self.__y_data = y_data

            # If a leaf, take average of y_data
            if self.__leaf and (leaf_type == 'regression'):
                self.__prediction = np.mean(y_data)
            elif self.__leaf and (leaf_type == 'classification'):
                temp_counter = Counter(y_data)
                self.__prediction = temp_counter.most_common(1)[0][0]
            else:
                self.__prediction = None

            # Set other values to None
            self.__lower_split = None
            self.__upper_split = None
            self.__col = None
            self.__split = None

        def is_leaf(self):
            """
            Return self.__leaf__
            :return: self.__leaf__ value
            """
            return self.__leaf

        def prune(self):
            """
            Prunes the node by setting the splits to none and making a leaf and eliminating unnecessary variables.
            """
            self.__leaf = True
            temp_counter = Counter(self.__y_data)
            self.__prediction = temp_counter.most_common(1)[0][0]
            # Set other values to None
            self.__lower_split = None
            self.__upper_split = None
            self.__col = None
            self.__split = None

        def set_split(self, idx, val):
            """
            Set the column/split index this node splits on.  Also
            sets the split value for a non-leaf node.
            :param idx: The index
            :param val: Specific value
            """
            self.__col = idx
            self.__split = val

        def set_lower_split_index(self, idx):
            """
            Set the lower split value.
            :param idx: the index of the lower split
            """
            self.__lower_split = idx

        def set_upper_split_index(self, idx):
            """
            Set the lower split value.
            :param idx: the index of the upper split
            """
            self.__upper_split = idx

        def get_x_data(self):
            """
            Return the x_data for this node (self.__data__)
            :return: self.__x_data
            """
            return self.__x_data

        def get_y_data(self):
            """
            Return the y_data for this node (self.__data__)
            :return: self.__y_data
            """
            return self.__y_data

        def get_prediction(self):
            """
            Return the prediction (if it is a leaf)
            :return: self.__split
            """
            return self.__prediction

        def get_col(self):
            """
            Get the column index the node splits on.
            :return: The column index
            """
            return self.__col

        def get_split(self):
            """
            Get the split value.
            :return: The split value
            """
            return self.__split

        def get_lower_split(self):
            """
            Get the value for index to the lower split data (if non-leaf node)
            :return: self.__lower_split
            """
            return self.__lower_split

        def get_upper_split(self):
            """
            Get the value for the index to the upper split data (if non-leaf node)
            :return: self.__upper_split
            """
            return self.__upper_split


class RegressionDecisionTree(_DecisionTree):
    """
    Regression Decision tree class.  Will inherit the decision tree class.
    Methods:
        Public
        ------
        Initialization: Initializes the class
        Private
        -------
        _rss: Calculates residual sum of squared error.
    """

    def __init__(self, split_type='rss', leaf_terminate=1):
        """
        Initialize the decision tree.
        :param split_type: the criterion to split a node (either rss, gini, gain_ratio)
        :param leaf_terminate: the type of decision tree (classification or regression)
        """
        if split_type == 'rss':
            split_func = self._rss
        else:
            split_func = self._rss
        # Initialize the super class
        super().__init__(
            'regression',
            split_func,
            leaf_terminate,
            False,
            False,
            (np.min, np.argmin)
        )

    def _rss(self, level, n, idx, split_val):
        """
        Calculates the residual sum of square errors for a specific region.
        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for
        :param split_val: The value to split on
        :return: The RSS
        """
        _, lower_y_data, _, upper_y_data = self._split_data(level, n, idx, split_val)
        return np.sum((lower_y_data - np.mean(lower_y_data))**2) + np.sum((upper_y_data - np.mean(upper_y_data))**2)


class ClassificationDecisionTree(_DecisionTree):
    """
    Classification Decision tree class.  Will inherit the decision tree class.
    NOTE: If there is a class tie, this module WILL PICK the lowest class.
    Methods:
        Public
        ------
        Initialization: Initializes the class
        Private
        -------
        _gini_impurity: Calculates the gini impurity
        _split_information: Calculates the split inforamtion to reduce continuous attributes
        _gini_impurity_gain: Calculate the gini impurity gain
        _gain_ratio: Calculate the gain ratio
        _expected_error: Gets the upper bound error based upon a 95% confidence interval
        _prune_tree: Goes through the pruning process to avoide overfitting
    """

    def __init__(self, split_type='gini', terminate='leaf', leaf_terminate=1, prune=False):
        """
        Initialize the decision tree.
        :param leaf_terminate: the amount of collections needed to terminate the tree with a leaf (defaults to 1)
        :param terminate: the way to terminate the classification tree (leaf/pure)
        :param split_type: the criteria to split on (gini/rss/gain_ratio)
        :param prune: whether we should use pessimistic pruning on the tree
        """
        # Initialize the split function
        if split_type == 'gain_ratio':
            split_func = self._gain_ratio
        else:
            split_func = self._gini_impurity_gain

        # Initialize termination criteria
        if terminate == 'leaf':
            if (leaf_terminate is None) or leaf_terminate < 1:
                raise ValueError('Cannot have non-positive termination criteria for terminate == "leaf"')
            leaf_terminate = leaf_terminate
            pure_terminate = False
        else:
            leaf_terminate = None
            pure_terminate = True

        super().__init__(
            'classification',
            split_func,
            leaf_terminate,
            pure_terminate,
            prune,
            (np.max, np.argmax)
        )

    @staticmethod
    def _gini_impurity(y_data):
        """
        Calculate the gini impurity (1 - sum(p(i)^2)
        :param y_data: the y data
        :return: the impurity
        """
        _, counts = np.unique(y_data, return_counts=True)
        return 1 - np.sum((counts / np.sum(counts))**2)

    @staticmethod
    def _split_information(x):
        """
        Calculate the split information (-sum(|S_i|/|S| * log_2(|S_i|/|S|))
        :param x: the specific x vector
        :return: the split information for that variable
        """
        c = Counter(x)
        freq = np.array([v for k, v in sorted(c.items())]) / len(x)
        return np.sum([-1 * i * np.log2(i) for i in freq if i > 0.0])

    def _gini_impurity_gain(self, level, n, idx, split_val):
        """
        Calculates the gain in gini impurity for a specific region.
        Should ONLY be used in classification problems.
        Gain = Curr Gini * Size - sum_{new nodes}(new_gini * size)
        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for
        :param split_val: The value to split on
        :return: The gini impurity for this split
        """
        y_data = self._node_list[level][n].get_y_data()
        _, lower_y_data, _, upper_y_data = self._split_data(level, n, idx, split_val)
        curr = self._gini_impurity(y_data)*len(y_data)
        lower = self._gini_impurity(lower_y_data)*len(lower_y_data)
        upper = self._gini_impurity(upper_y_data)*len(upper_y_data)
        return curr - (lower + upper)

    def _gain_ratio(self, level, n, idx, split_val):
        """
        Calculates the gain ratio, which is equal to the (impurity gain)/(split information)
        Should ONLY be used in classification problems.
        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for
        :param split_val: The value to split on
        :return: The gain ratio
        """
        x_data = self._node_list[level][n].get_x_data()
        return self._gini_impurity_gain(level, n, idx, split_val) / self._split_information(x_data[:, idx])

    @staticmethod
    def _expected_error(y_data):
        """
        Calculate the expected error using a 95 percent CI, and approximating the result as a normal
        approximation to a binomial distribution.
        :param y_data: the y_data for this leaf
        :return: the error from a 95 percent ci
        """
        # Get the bincount, and the error probability
        freq = np.bincount(y_data) / len(y_data)
        p = 1 - np.max(freq)
        # Get the upper bound
        return (p + 1.96 * np.sqrt((p * (1 - p)) / len(y_data))) * len(y_data)

    def _prune_tree(self, curr_idx=None):
        """
        Prune the tree using the expected error.  Will recursively iterate through parent nodes and prune
        the tree if necessary.
        :param curr_idx: The level and index of the current node
        """
        level, n = curr_idx[0], curr_idx[1]
        # Check if leaf
        if self._node_list[level][n].is_leaf():
            pass
        else:
            # Check whether to prune the upper and lower branches
            lower_idx = [curr_idx[0] + 1, self._node_list[curr_idx[0]][curr_idx[1]].get_lower_split()]
            self._prune_tree(lower_idx)
            upper_idx = [curr_idx[0] + 1, self._node_list[curr_idx[0]][curr_idx[1]].get_upper_split()]
            self._prune_tree(upper_idx)
            # Check whether to prune this branch
            curr_error = self._expected_error(self._node_list[curr_idx[0]][curr_idx[1]].get_y_data())
            lower_error = self._expected_error(self._node_list[lower_idx[0]][lower_idx[1]].get_y_data())
            upper_error = self._expected_error(self._node_list[upper_idx[0]][upper_idx[1]].get_y_data())
            # Test whether to prune
            if curr_error < (lower_error + upper_error):
                self._node_list[curr_idx[0]][curr_idx[1]].prune()
                self._node_list[lower_idx[0]][lower_idx[1]] = None
                self._node_list[upper_idx[0]][upper_idx[1]] = None

                # Delete list if no nodes are left
                if set(self._node_list[curr_idx[0] + 1]) == {None}:
                    del self._node_list[curr_idx[0] + 1]
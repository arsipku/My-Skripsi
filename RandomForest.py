"""
RandomForest.py
Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/
Self-implementation of a random forest.
Package requirements:
numpy
pandas
DecisionTree.py (self implementation)
"""

# Imports
import numpy as np
import DecisionTree as DT
from collections import Counter


class RandomForest(object):
    """
    Abstract decision tree class.  Will use a linked-list-esque implementation where node is a "Node" class,
    and the node holds a cutoff, and then reference to another node.  Nodes can also hold a terminating value.
    Abstraction is made using the private (_) to not be able to implement outside.
    Parameters:
    self._samp_size: Size of sample to take for fitting tree
    self._num_trees: The number of trees to create
    self._num_features: The number of features to use per tree (defaults to None which uses all features)
    self._split_type: whether to use gini vs. gain ratio etc
    self._terminate: The termination criteria (pure/not)
    self._leaf_terminate: The number of samples in a leaf (not used if self._teriminate = pure)
    self._oob_flag: The flag to calculate the oob error
    self._trees: The list of trees
    self._cols_used: The columns used to fit a tree
    self._oob_errors: The oob error per tree
    Methods:
        Public
        ------
        Initialization: Initializes the class
        fit: Fit the x data and y data
        get_oob_error: Get the out of bag error
        predict: Predict the values for a new dataset
        get_trees: Get the list of decision trees
        get_cols_used: Get the columns used for each tree
        Private
        -------
        __check_feature_size: Checks to make sure feature size is less than proposed number of features
        __get_sample: Take a random sample of the x/y data to produce a train/out of bad set
        __get_tree: Creates a new classification decision tree
        __calculate_oob_error: Calculates the out of back error for a specific sample
    """

    def __init__(
            self,
            samp_size=0.5,
            num_trees=10,
            num_features=None,
            split_type='gini',
            terminate='leaf',
            leaf_terminate=1,
            oob=False
    ):
        """
        Initialize the RandomForest class.
        :param samp_size: The number of samples to put within each decision tree
        :param num_trees: The number of trees to make
        :param split_type: The criteria to split on
        :param terminate: The termination criteria
        :param leaf_terminate: The number of samples to put into a leaf
        :param oob: Whether to cross-validated using an out-of-bag sample
        """
        # Set parameters
        self._samp_size = samp_size
        self._num_trees = num_trees
        self._num_features = num_features
        self._split_type = split_type
        self._terminate = terminate
        self._leaf_terminate = leaf_terminate
        self._oob_flag = oob
        self._trees = []
        self._cols_used = []
        self._oob_errors = []

    def fit(self, x_data, y_data):
        """
        Fit (train) a Random Forest model to the data.
        :param x_data: The dataset to train the decision tree with
        :param y_data: The result vector we are classifying (target)
        """
        # handle the data
        x_data, y_data = DT.ClassificationDecisionTree.handle_data(x_data, y_data)
        # Check feature size
        if self._num_features is None:
            self._num_features = x_data.shape[1]
        # Make the number of trees determinate by self._num_trees
        for i in range(self._num_trees):
            # Get tree
            x_in, y_in, x_out, y_out = self.__get_sample(x_data, y_data)
            cdt = self.__get_tree(x_in, y_in)
            # Calculate oob if necessary
            if self._oob_flag:
                self._oob_errors.append(self.__calculate_oob_error(cdt, x_out, y_out))
            self._trees.append(cdt)

    def __check_feature_size(self, x_data):
        """
        Check to make sure the feature size is <= x_data. If self._num_features is none, will set
        self._num_features to x_data
        :param x_data: The x_data matrix to fit to
        """
        # Check if self._num_features is none, if so set to shape
        if self._num_features is None:
            self._num_features = x_data.shape[1]
        if self._num_features > x_data.shape[1]:
            raise ValueError('Number of features is greater than given X features.')

    def __get_sample(self, x, y):
        """
        Get a sample from two indices.  Will also sample num features if necessary
        :param x: The x data to sample from
        :param y: The y data to sample from
        :return: The sampled x data, y data and the out of sample x data/y data
        """
        # Take the random sample
        idx = np.random.choice(len(y), size=int(np.floor(self._samp_size * len(y))), replace=True)
        # Sample features
        cols = np.random.choice(x.shape[1], size=self._num_features, replace=False)
        self._cols_used.append(cols)

        # Get the sample splits
        mask = np.ones(len(y), dtype=bool)
        mask[idx] = False
        x_in = x[idx, :]
        x_in = x_in[:, cols]
        y_in = y[idx]
        x_out = x[mask, :]
        x_out = x_out[:, cols]
        y_out = y[mask]

        return x_in, y_in, x_out, y_out

    def __get_tree(self, x, y):
        """
        Create a decision tree based upon self._num_trees.
        :param x: The x data to fit to (input)
        :paray y: The y data to fit to (target)
        :return: A new CDT
        """
        dt = DT.ClassificationDecisionTree(self._split_type, self._terminate,  self._leaf_terminate, prune=False)
        dt.fit(x, y)
        return dt

    @staticmethod
    def __calculate_oob_error(cdt, x_out, y_out):
        """
        Calculate the oob error for a tree by predicting on the out of bag sample.
        :param cdt: The fit decision tree
        :param x_out: The out of bag input
        :param y_out: THe out of bad target
        """
        y_pred = cdt.predict(x_out)
        return (np.mean(y_out) - np.mean(y_pred))**2

    def get_oob_error(self):
        return np.mean(self._oob_errors)

    def predict(self, x_data):
        """
        Predict the y (target) for this x_data
        :param x_data: The data to predict off of
        :return: The predicted target data (y)
        """
        # Handle data
        x_data, _ = DT.ClassificationDecisionTree.handle_data(x_data)
        preds = []
        # Predict on each tree
        for i in range(len(self._trees)):
            preds.append(self._trees[i].predict(x_data[:, self._cols_used[i]]))
        pred = np.column_stack(preds)
        return np.array([Counter(pred[i, :]).most_common(1)[0][0] for i in range(pred.shape[0])])

    def get_trees(self):
        """
        Get the list of trees.
        :return: The list of trees.
        """
        return self._trees

    def get_cols_used(self):
        """
        Get the columns used for each tree.
        :return: The columns used
        """
        return self._cols_used
import numpy as np
import math
from numba import cuda, jit

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

##################################################################################################

@cuda.jit
def bincount_kernel(y, hist):
    index = cuda.grid(1)
    if index < y.size:
        cuda.atomic.add(hist, y[index], 1)

@cuda.jit
def entropy_kernel(hist, out, n):
    index = cuda.grid(1)
    num_bins = hist.shape[0]
    if index < num_bins:
        ps = hist[index] / n
        if ps > 0:
            entropy = -ps * math.log2(ps)
        cuda.atomic.add(out, 0, entropy)

@cuda.jit
def split_kernel(X_column, split_thresh, left_idxs, right_idxs):
    i = cuda.grid(1)
    if i < X_column.shape[0]:
        if X_column[i] <= split_thresh:
            left_idxs[i] = i
            right_idxs[i] = -1
        else:
            right_idxs[i] = i
            left_idxs[i] = -1

@jit(nopython=True)
def filter_idxs(left_idxs, right_idxs):
    left_idxs = left_idxs[(left_idxs >= 0) & (left_idxs <= 42000)]
    right_idxs = right_idxs[(right_idxs >= 0) & (right_idxs <= 42000)]
    return left_idxs, right_idxs

class DecisionTreeParallel:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None    
        
    def entropy(self, y):
        hist = cuda.to_device(np.zeros(10, dtype=np.int32))
        out = cuda.to_device(np.zeros(1, dtype=np.float64))
        d_y = cuda.to_device(y)
        n = y.shape[0]
        block_size = 256
        grid_size = math.ceil(y.shape[0] / block_size)
        bincount_kernel[grid_size, block_size](d_y, hist)
        entropy_kernel[grid_size, block_size](hist, out, n)
        cuda.synchronize()
        entropy = out.copy_to_host()[0]
        return entropy

    def split(self, X_column, split_thresh):
        d_left_idxs = cuda.device_array(X_column.shape[0], dtype=np.int32)
        d_right_idxs = cuda.device_array(X_column.shape[0], dtype=np.int32)
        d_X_column = cuda.to_device(X_column)
        block_size = 64
        grid_size = math.ceil(X_column.shape[0] / block_size)
        split_kernel[grid_size, block_size](d_X_column, split_thresh, d_left_idxs, d_right_idxs)
        cuda.synchronize()
        left_idxs = np.empty(X_column.shape[0], dtype=np.int32)
        right_idxs = np.empty(X_column.shape[0], dtype=np.int32)
        left_idxs = d_left_idxs.copy_to_host()
        right_idxs = d_right_idxs.copy_to_host()
        left_idxs, right_idxs = filter_idxs(left_idxs, right_idxs)
        return left_idxs, right_idxs

    def information_gain(self, y, X_column, threshold):
        parent_entropy = self.entropy(y)
        left_idxs, right_idxs = self.split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        information_gain = parent_entropy - child_entropy
        return information_gain

    def best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self.information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

    def most_common_label(self, y):
        y_array = np.array(y)
        unique, counts = np.unique(y_array, return_counts=True)
        index = np.argmax(counts)
        return unique[index]

    def grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self.best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self.split(X[:, best_feature], best_thresh)
        left = self.grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self.grow_tree(X[right_idxs], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self.grow_tree(X, y)

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

##################################################################################################

class RandomForestParallel:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeParallel(max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features)
            X_sample, y_sample = self.bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def most_common_label(self, y):
        y_array = np.array(y)
        unique, counts = np.unique(y_array, return_counts=True)
        index = np.argmax(counts)
        return unique[index]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common_label(pred) for pred in tree_preds])
        return predictions
import numpy as np
import math
from numba import cuda, jit, int32

##########################################################################################################

def unique_values(y):
    unique = []
    for value in y:
        if value not in unique:
            unique.append(value)
    unique.sort()
    return unique

@cuda.jit
def compute_hist_kernel(in_array, hist, nBins):
    s_hist = cuda.shared.array(shape=10, dtype=int32)
    for bin in range(cuda.threadIdx.x, nBins, cuda.blockDim.x):
        s_hist[bin] = 0
    cuda.syncthreads()

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < in_array.shape[0]:
        cuda.atomic.add(s_hist, in_array[i], 1)
    cuda.syncthreads()

    for bin in range(cuda.threadIdx.x, nBins, cuda.blockDim.x):
        cuda.atomic.add(hist, bin, s_hist[bin])
    cuda.syncthreads()

@cuda.jit
def entropy_kernel(hist, out, n):
    shared_hist = cuda.shared.array(shape=10, dtype=int32)
    
    if cuda.threadIdx.x < hist.shape[0]:
        shared_hist[cuda.threadIdx.x] = hist[cuda.threadIdx.x]
    cuda.syncthreads()

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < hist.shape[0]:
        ps = shared_hist[i] / n
        if ps > 0:
            entropy = -ps * math.log2(ps)
        else:
            entropy = 0
        cuda.atomic.add(out, 0, entropy)
    cuda.syncthreads()

def entropy(y):
    nBins = 10
    hist = np.zeros(10, dtype=np.int32)
    out = np.zeros(1, dtype=np.float64)
    d_hist = cuda.to_device(hist)
    d_in = cuda.to_device(y)
    d_out = cuda.to_device(out)
    n = y.shape[0]
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    compute_hist_kernel[grid_size, block_size](d_in, d_hist, nBins)
    cuda.synchronize()
    entropy_kernel[grid_size, block_size](d_hist, d_out, n)
    cuda.synchronize()
    entropy = d_out.copy_to_host()[0]
    return entropy

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

@jit(parallel=True, cache=True)
def filter_idxs(left_idxs, right_idxs):
    left_idxs = left_idxs[(left_idxs >= 0)]
    right_idxs = right_idxs[(right_idxs >= 0)]
    return left_idxs, right_idxs

def split(X_column, split_thresh):
    X_column = np.ascontiguousarray(X_column)
    d_left_idxs = cuda.device_array(X_column.shape[0], dtype=np.int32)
    d_right_idxs = cuda.device_array(X_column.shape[0], dtype=np.int32)
    d_X_column = cuda.to_device(X_column)
    block_size = 256
    grid_size = math.ceil(X_column.shape[0] / block_size)
    split_kernel[grid_size, block_size](d_X_column, split_thresh, d_left_idxs, d_right_idxs)
    cuda.synchronize()
    left_idxs = np.empty(X_column.shape[0], dtype=np.int32)
    right_idxs = np.empty(X_column.shape[0], dtype=np.int32)
    left_idxs = d_left_idxs.copy_to_host()
    right_idxs = d_right_idxs.copy_to_host()
    left_idxs, right_idxs = filter_idxs(left_idxs, right_idxs)
    return left_idxs, right_idxs

def most_common_label(y):
    unique_labels = list(set(y))
    counts = [0] * len(unique_labels)
    for label in y:
        for i in range(len(unique_labels)):
            if unique_labels[i] == label:
                counts[i] += 1
                break
    most_common_index = 0
    max_count = counts[0]
    for i in range(1, len(counts)):
        if counts[i] > max_count:
            most_common_index = i
            max_count = counts[i]
    most_common = unique_labels[most_common_index]
    return most_common

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]

##########################################################################################################

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

##########################################################################################################

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)
        left_idxs, right_idxs = split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = unique_values(X_column)
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(unique_values(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = most_common_label(y)
            return Node(value=leaf_value)
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self.best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = split(X[:, best_feat], best_thresh)
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self.grow_tree(X, y)
    
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    @cuda.jit
    def fit_kernel(list_X_samp, list_y_samp, tree_list, n_trees, min_samples_split, max_depth, n_feats):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < n_trees:
            tree = DecisionTree(min_samples_split=min_samples_split, max_depth=max_depth, n_feats=n_feats)
            X_samp = list_X_samp[i]
            y_samp = list_y_samp[i]
            tree.fit(X_samp, y_samp)
            cuda.atomic.add(tree_list, i, tree)

    def fit(self, X, y):
        self.trees = []
        list_X_samp = []
        list_y_samp = []
        for i in range(self.n_trees):
            X_samp, y_samp = bootstrap_sample(X, y)
            list_X_samp.append(X_samp)
            list_y_samp.append(y_samp)
        
        d_list_X_samp = cuda.to_device(list_X_samp)
        d_list_y_samp = cuda.to_device(list_y_samp)
        d_trees = cuda.device_array(self.n_trees, dtype=types.pyobject)

        block_size = 256
        grid_size = math.ceil(self.n_trees / block_size)
        fit_kernel[grid_size, block_size](d_list_X_samp, d_list_y_samp, d_trees, self.n_trees, self.min_samples_split, self.max_depth, self.n_feats)
        cuda.synchronize()

        self.trees = d_trees.copy_to_host()

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
from keras_sdec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np
import scipy.sparse as sp
import os
def get_mnist():
    np.random.seed(1234)

    data = np.load("usps.npz")
    x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    x_all = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    X = x_all.reshape(-1, x_all.shape[1] * x_all.shape[2])

    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    Y = Y[p]+1

    Y_temp = Y.reshape(-1, 1)
    A = np.where(Y_temp.dot(Y_temp.T) == np.square(Y_temp), 1, 0)
    A = preprocess_graph(A)

    return X, Y, A

def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.toarray()

if not os.path.exists('data_usps.npz'):
    X, Y, A  = get_mnist()
    AX = np.dot(A,X)
    np.savez('data_usps.npz', X=X, Y=Y, AX=AX)
else:
    data = np.load('data_usps.npz')
    X = data['X']
    Y = data['Y']
    AX = data['AX']

c = DeepEmbeddingClustering(n_clusters=10, input_dim=256, pretrained_weights='pretrained_weights')
c.initialize(X,  finetune_iters=100000, layerwise_pretrain_iters=50000)
c.cluster(X, AX, y=Y, iter_max=1e5)
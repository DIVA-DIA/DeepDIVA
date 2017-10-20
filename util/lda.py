"""
Linear Discriminant Analysis Algorithm

It computes both the transformation matrix and the linear discriminants
"""

import logging

import numpy as np

"""
X = [[4, 1],
     [2, 4],
     [2, 3],
     [3, 6],
     [4, 4],
     [9, 10],
     [6, 8],
     [9, 5],
     [8, 7],
     [10, 8]]
X = np.array(X)

y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y = np.array(y)
"""


def transform(X, y):
    """
    Computes the transformation matrix
    :param X: np.array(N,m)
        input data. On the rows are the samples and on the columns are the
        dimensions. Typically you want to have NUM_ROWS >> NUM_COLS otherwise
        you get a singular-matrix error
    :param y: np.array(N,)
        labels of the input data in the range [0,num_classes-1].
    :return:
        L=np.array(N,m) = the LDA transformation matrix L
        C=np.array(N,) = a bias vector with zeros (in LDA it must be so!)
    """
    # Check for sizes
    assert len(X.shape) == 2
    assert len(y.shape) == 1

    # Detect number of unique classes
    NUM_CLASSES = len(np.unique(y))

    ###########################################################################
    # Step 1: Computing the mean vectors
    mean_vectors = []
    for cl in range(NUM_CLASSES):
        mean_vectors.append(np.mean(X[y == cl], axis=0))
        # print('Mean Vector class %s: %s' % (cl, mean_vectors[cl]))

    overall_mean = np.mean(X, axis=0)

    nmu = len(X) / NUM_CLASSES

    ###########################################################################
    # Step 2: Computing the Scatter Matrices
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for cl in range(NUM_CLASSES):
        S_W += (np.transpose(X[y == cl] - mean_vectors[cl]).dot(
            X[y == cl] - mean_vectors[cl])) / len(X[y == cl])
    S_W *= nmu
    # print('within-class Scatter Matrix:\n', S_W)

    S_B = np.zeros((X.shape[1], X.shape[1]))
    for cl in range(NUM_CLASSES):
        S_B += (np.transpose(
            np.expand_dims(mean_vectors[cl] - overall_mean, 0)).dot(
            np.expand_dims(mean_vectors[cl] - overall_mean, 0))) / len(
            X[y == cl])
    S_B *= nmu
    # print('between-class Scatter Matrix:\n', S_B)

    ###########################################################################
    # Step 3: Solving the generalized eigenvalue problem
    # beware of pinv() instead of inv()!!
    try:
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        pass

    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    # continue with what you were doing

    # eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    ###########################################################################
    # Step 4: Selecting linear discriminants for the new feature subspace
    # Make a  list of(eigenvalue, eigenvector)tuples
    # eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    # eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # 4.2.Choosing k eigenvectors with the largest eigenvalues
    # L = np.hstack((eig_pairs[0][1].reshape(X.shape[1], 1), eig_pairs[1][1].reshape(X.shape[1], 1)))
    # print('Matrix L:\n', L.real)
    L = eig_vecs[:, np.argsort(eig_vals)[::-1]]

    # Biases are zero in LDA transform
    B = np.zeros(X.shape[1])

    # We return the transpose because PYTORCH WANT THE MATRIX to be flipped!! BE CAREFUL WHEN THINKING ABOUT THIS!
    return L.T, B.T


def discriminants(X, y):

    # Check for sizes
    assert len(X.shape) == 2
    assert len(y.shape) == 1

    # Compute linear discriminants
    logging.debug('Compute linear discriminants')
    NUM_CLASSES = len(np.unique(y))
    pooled_conv = np.zeros((X.shape[1], X.shape[1]))

    # Step 1: Computing the mean vectors
    logging.debug('Step 1: Computing the mean vectors')
    for cl in range(NUM_CLASSES):
        pooled_conv += (float(len(X[y == cl]) - 1) / (
        len(X) - NUM_CLASSES)) * np.cov(np.transpose(X[y == cl]))

    # print('Pooled_cov: \n{}'.format(pooled_conv))

    # Step 2: Computing prior probabilities
    logging.debug('Step 2: Computing prior probabilities')
    prior_prob = []
    for cl in range(NUM_CLASSES):
        prior_prob.append(float(len(X[y == cl])) / len(X))

    # Step 3: Main routine
    logging.debug('Step 3: Main routine')
    W = np.zeros((NUM_CLASSES, X.shape[1]))
    C = np.zeros(NUM_CLASSES)
    for cl in range(NUM_CLASSES):
        logging.debug('Class {}'.format(cl))
        mean_vector = np.mean(X[y == cl], axis=0)
        W[cl] = np.linalg.lstsq(pooled_conv, mean_vector)[0]
        C[cl] = -0.5 * np.matmul(W[cl], np.expand_dims(mean_vector, 0).T) + np.log(prior_prob[cl])

    W = W.T
    C = C.T

    logging.debug('Finish')
    # We return the transpose because PYTORCH WANT THE MATRIX to be flipped!! BE CAREFUL WHEN THINKING ABOUT THIS!
    return W.T, C.T

    """
    L = np.matmul(X,W)+ C
    sum = np.zeros(len(X))
    P = np.zeros((len(X),NUM_CLASSES))
    for i in range(len(X)):
        for cl in range(NUM_CLASSES):
            sum[i] += np.exp(L[i,cl])
    
        for cl in range(NUM_CLASSES):
            P[i,cl] = np.exp(L[i,cl]) / sum[i]
    """

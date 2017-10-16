"""
Here are defined the basic methods to initialize a CNN in a data driven way.
For initializing complex architecture or using more articulated stuff (e.g LDA
has two functions) one should implement his own init function.
"""

# Utils
import logging

import numpy as np
import torch

# Init tools
import util.lda as lda


def init(model, data, *args, **kwargs):
    """
    Initialize a standard CNN composed by convolutional layer followed by fully
    connected layers.
    :param model:
        the network to initialize
    :param data:
        the dataloader to take the data from
    :param args:
        parameters for the function. In particular:
            > num_points: integer
            specifies how many points should be used to compute the data-driven
            initialization
    :param kwargs:
        parameters for the function.
    :return:
        nothing, the parameters of the network are modified in place
    """
    ###############################################################################################
    # Collect initial data
    logging.debug('Collect initial data')
    X = []
    y = []
    for i, (input, target) in enumerate(data, 1):
        X.append(torch.autograd.Variable(input))
        y.append(torch.autograd.Variable(target))
        if i * data.batch_size >= kwargs['num_points']:
            break

    # TODO select the input patches of the right size of the filter of 1st layer

    ###############################################################################################
    # Compute first layer param
    logging.info('Compute first layer param')
    # W, C = lda.transform(
    #     X=minibatches_to_matrix(X),
    #     y=np.squeeze(minibatches_to_matrix(y))
    # )

    W = np.zeros((16, 16))
    B = np.zeros(16)

    ###############################################################################################
    # Iterate over all layers
    logging.info('Iterate over all layers')
    for index, module in enumerate(model.children()):
        logging.info('Layer: {}'.format(index))

        # Assign parameters
        logging.info('Assign parameters')
        # TODO select from LDA the relevant columns for as many filters there are
        module[0].weight.data = torch.Tensor(W)
        module[0].bias.data = torch.Tensor(B)

        # If the layer is not convolutional then flatten the data because
        # we assume it is a fully connected one
        if 'conv' not in str(type(list(module.children())[0])):
            logging.info('Flattening input')
            X = X.view(X.size(0), -1)

        # Forward pass
        logging.info('Forward pass')
        for i, minibatch in enumerate(X):
            X[i] = module(X[i])

        # Compute data-driven parameters
        W, C = lda.transform(
            X=minibatches_to_matrix(X),
            y=np.squeeze(minibatches_to_matrix(y))
        )


def minibatches_to_matrix(X):
    return np.array([s.data.view(-1).numpy() for m in X for s in m])


"""
def fuck_i_deleted_something_here():
    data = [item for item in train_loader]
    X, y = np.vstack([item[0].numpy() for item in data]), np.concatenate([item[1].numpy() for item in data])
    del data
    X, y = X[:num_points_lda], y[:num_points_lda]
    le.fit(y)

    # Compute LDA for 1st Conv Layer
    X_5x5 = X[:, :, 11 - 2:11 + 3, 11 - 2:11 + 3]
    X_flat = X_5x5.reshape(X_5x5.shape[0], np.prod(np.array(X_5x5.shape[1:])))
    L = transform(X_flat, le.transform(y)).T
    weights = model.conv1.weight.data.numpy()
    for i in range(len(weights)):
        weights[i] = L[:, i].reshape(3, 5, 5)
    model.conv1.weight.data = torch.from_numpy(weights).float()
    model.conv1.bias.data = torch.from_numpy(np.zeros(model.conv1.bias.data.size())).float()

    outputs = []
    for i in range(0, len(X), args.batch_size):
        outputs.append(model.ss(model.conv1(V(torch.from_numpy(X[i:i + args.batch_size]).float()))))
        # outputs.append([model(torch.autograd.Variable(torch.from_numpy(X[i:i + args.batch_size]).float()))[1][2]])

    X_3x3 = np.vstack([item.data.numpy() for item in outputs])[:, :, 3 - 1:3 + 2, 3 - 1:3 + 2]

    # X_3x3 = model.conv1(V(torch.from_numpy(X).float())).data.numpy()[:,:,3-1:3+2,3-1:3+2]
    X_flat = X_3x3.reshape(X_3x3.shape[0], np.prod(np.array(X_3x3.shape[1:])))
    L = transform(X_flat, le.transform(y)).T
    weights = model.conv2.weight.data.numpy()
    for i in range(len(weights)):
        weights[i] = L[:, i].reshape(24, 3, 3)
    model.conv2.weight.data = torch.from_numpy(weights).float()
    model.conv2.bias.data = torch.from_numpy(np.zeros(model.conv2.bias.data.size())).float()

    outputs = []
    for i in range(0, len(X), args.batch_size):
        outputs.append(
            model.ss(model.conv2(model.ss(model.conv1(V(torch.from_numpy(X[i:i + args.batch_size]).float()))))))
        # outputs.append([model(torch.autograd.Variable(torch.from_numpy(X[i:i + args.batch_size]).float()))[1][1]])
    X_3x3 = np.vstack([item.data.numpy() for item in outputs])  # [:,:,3-1:3+2,3-1:3+2]

    # X_3x3 = model.conv2(model.conv1(V(torch.from_numpy(X).float()))).data.numpy()
    X_flat = X_3x3.reshape(X_3x3.shape[0], np.prod(np.array(X_3x3.shape[1:])))
    L = transform(X_flat, le.transform(y)).T
    weights = model.conv3.weight.data.numpy()
    for i in range(len(weights)):
        weights[i] = L[:, i].reshape(48, 3, 3)
    model.conv3.weight.data = torch.from_numpy(weights).float()
    model.conv3.bias.data = torch.from_numpy(np.zeros(model.conv3.bias.data.size())).float()

    outputs = []
    for i in range(0, len(X), args.batch_size):
        outputs.append(model.ss(model.conv3(
            model.ss(model.conv2(model.ss(model.conv1(V(torch.from_numpy(X[i:i + args.batch_size]).float()))))))))
        # outputs.append([model(torch.autograd.Variable(torch.from_numpy(X[i:i + args.batch_size]).float()))[0][0]])

    X_fc = np.squeeze(np.vstack([item.data.numpy() for item in outputs]))
    # X_fc = model.fc(model.conv3(model.conv2(model.conv1(V(torch.from_numpy(X).float())))).view(len(X),-1)).data.numpy()

    W, C = discriminants(X_fc, le.transform(y))
    # import pdb; pdb.set_trace()
    W = W.T
    C = C.T
    weights = np.zeros(model.fc.weight.data.numpy().shape)
    bias = np.zeros(model.fc.bias.data.numpy().shape)

    for i in range(len(W)):
        weights[i] = W[i, :]
    for i in range(len(C)):
        bias[i] = C[i]
    model.fc.weight.data = torch.from_numpy(weights).float()
    model.fc.bias.data = torch.from_numpy(bias).float()

    print("LDA initialization took {} seconds".format(time.time() - t))

    return model
    """

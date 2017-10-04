import torch


def init(model, data, *args, **kwargs):
    ###############################################################################################
    # Iterate over all layers
    layer = 0

    X, y = assign
    the
    data

    for module in model.modules():
        W, C = computeLDA(X, args.num_points)  # where args.init is ref to LDA(*args)
        X = forward(module, X)


def forward(layer, input):
    output = []
    for x in range(len(input)):
        output.append(layer.forward(input))
    return output


def initialize_model_lda(model, train_loader, num_points_lda):
    from torch.autograd import Variable as V
    # model = LDA_init_net(8)
    le = LabelEncoder()
    t = time.time()
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

def dataset_mnist():
    import pandas as pd
    df = pd.read_csv('mnist_train.csv',header=None)
    y = df.iloc[:,0].values
    x = (df.iloc[:,1:].values).astype("float32") /255
    return x, y

def dataset_cifar10():
    import pickle
    import numpy as np
    file = r'C:\work_env\vs_code\deep_learning\cifar-10-batches-py\data_batch_1'

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    for i in range(len(dict.get(b'data'))):
        image = np.array([[dict.get(b'data')[i][0:1024],dict.get(b'data')[i][1024:2048],dict.get(b'data')[i][2048:3072]]])
    print(image.shape)
    print((dict.get(b'data'))[0].shape)

dataset_cifar10()
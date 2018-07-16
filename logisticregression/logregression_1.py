import numpy as np
import time


def sigmoid(inx):
    return 1.0/(1 + np.exp(-inx))


def trainlogression(train_x, train_y, opts):
    """
    :param train_x: train dattaset
    :param train_y:train labelset
    :param opts: train param
    :return: weight
    """
    starttime = time.time()
    numsample, numfeature = np.shape(train_x)
    alpha = opts['alpha']
    maxlter = opts['maxlter']
    weights = np.ones((numfeature, 1))
    print(weights)

    for k in range(maxlter):
        if opts['optimizetype'] == 'gradDescent':
            output = sigmoid(train_x*weights)
            error = train_y-output
            weights = weights+alpha*train_x.transpose()*error
        elif opts['optimizetype'] == 'stocGradDescent':
            for i in range(numsample):
                output = sigmoid(train_x[i, :]*weights)
                error = train_y[i, 0]-output
                weights = weights+alpha*train_x[i, :].transpose()*error
        elif opts['optimizetype'] == 'smoothStocGradDescent':
            dataindex = range(numsample)
            for i in range(numsample):
                alpha = 4.0/(1.0+k+i)+0.01
                output = sigmoid(train_x[randindex, :]*weights)
                randindex = int(np.random.uniform(0, len(dataindex)))
                error = train_y[randindex, 0]-output
                weights = weights+alpha*train_x[randindex, :].transpose()*error
                del(dataindex[randindex])
        else:
            raise NameError('Not support optimize method type!')
    print('Congratulations,training complete!Took %fs!' % (time.time()-starttime))
    return weights


def testlogregression(weight, test_x, test_y):
    numsample, numfeature = np.shape(test_x)
    matchcount = 0
    for i in range(numsample):
        output = sigmoid(test_x[i, :]*weight)
        predict = (output >= 0.5)
        # [0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchcount += 1
    accuracy = float(matchcount)/numsample
    return accuracy


def loaddata():
    train_x = []
    train_y = []
    with open(r'E:\PycharmProjects\machine_learning\logisticregression\testset.txt') as f:
        for line in f.readlines():
            linearr = line.strip().split()
            train_x.append([float(linearr[0]), float(linearr[1])])
            train_y.append(float(linearr[2]))
        return np.mat(train_x), np.mat(train_y).transpose()


if __name__ == '__main__':
    train_xx, train_yy = loaddata()
    test_xx, test_yy = train_xx, train_yy
    opt = {'alpha': 0.01, 'maxlter': 25, 'optimizetype': 'gradDescent'}
    optimalweights = trainlogression(train_xx, train_yy, opt)
    accuracy_rate = testlogregression(optimalweights, test_xx, test_yy)
    print(accuracy_rate)

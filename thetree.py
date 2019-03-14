from math import log


def calcsnannonent(dataset):
    # 计算数据信息熵
    numentries = len(dataset)
    labelcount = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelcount.keys():
            labelcount[currentlabel] = 0
        labelcount[currentlabel] += 1
    shannonent = 0.0
    for key in labelcount:
        prob = float(labelcount[key])/numentries
        shannonent -= prob*log(prob, 2)
    return shannonent


def creatdataset():
    # 创建数据
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no', 'yes']
    return dataset, labels


def splitdataset(dataset, axis, value):
    # 按照指定特征划分数据集，axis是特征，value是返回特征的值。单独抽取出来
    retdata = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedFeatvec = featvec[:axis]
            reducedFeatvec.extend(featvec[axis+1:])
            retdata.append(reducedFeatvec)
    return retdata


myData, labels = creatdataset()
ends = calcsnannonent(myData)
retdata = splitdataset(myData, 0, 0)
print(myData, '\n', labels, '\n', ends, '\n', retdata)




# -*- coding: utf-8 -*-

import os

import time

import numpy as np

from math import log

from sklearn.externals import joblib


def trainNaiveBayesClassifier():
    """
    训练朴素贝叶斯分类器
    """
    # 训练集特征矩阵保存路径
    train_matrix_path = 'matrix/train/matrix.pkl'
    # 后验概率保存路径
    prob_clf_path = 'classifier/baseline.pkl'
    
    # 加载数据
    matrix = joblib.load(train_matrix_path)

    # 计算每一类中每个特征的后验概率
    prob_clf = []
    for clf in range(0, 10):
        start_row = clf * 50000
        end_row = (clf+1) * 50000

        vector = matrix[start_row:end_row].sum(axis=0)
        total_word = matrix[start_row:end_row].sum()

        prob = np.log((vector+1)/float(total_word))
        prob_clf.append(prob)

    # 保存后验概率
    joblib.dump(prob_clf, prob_clf_path)


def testNaiveBayesClassifier():
    """
    测试朴素贝叶斯分类器
    """
    # 测试集特征矩阵保存路径
    test_matrix_path = 'matrix/test/matrix.pkl'
    # 后验概率保存路径
    prob_clf_path = 'classifier/baseline.pkl'

    # 加载数据
    matrix = joblib.load(test_matrix_path)
    target = np.array([x for x in range(10) for i in range(50000)])

    # 加载贝叶斯每一类的后验概率
    prob_clf = joblib.load(prob_clf_path)

    # 预测
    confusion_matrix = np.zeros(shape=(10,10),dtype=int)
    for i in range(0, len(target)):
        max_value, predicted = -float('inf'), 0
        a = np.array(matrix[i].sum(axis=0))[0]

        for clf in range(0, 10):
            b = np.array(prob_clf[clf])[0]
            value = np.dot(a, b)
            if value > max_value:
                max_value = value
                predicted = clf

        confusion_matrix[target[i]][predicted] += 1

    joblib.dump(confusion_matrix, 'results/Bayes_confusion_matrix.pkl')

    # 统计
    recall_list, precision_list, f_list = [], [], []
    correct = 0
    r = confusion_matrix.sum(axis=1)
    p = confusion_matrix.sum(axis=0)
    for clf in range(0, 10):
        recall = confusion_matrix[clf][clf] / float(r[clf])
        precision = confusion_matrix[clf][clf] / float(p[clf])
        f = 2*recall*precision/(recall+precision)
        recall_list.append(recall)
        precision_list.append(precision)
        f_list.append(f)
        correct += confusion_matrix[clf][clf]
    correct /= float(matrix.shape[0])

    # 打印测试报告
    print confusion_matrix,'\n'
    print '{0:>14}\t{1:<10}\t{2:<10}\t{3:<10}'.format('classification','Recall','Precision','F1-Score')
    for i, target_name in enumerate(os.listdir('data/test/raw/')):
        print '{0:>14}\t{1:<10.4f}\t{2:<10.4f}\t{3:<10.4f}'.format(target_name, recall_list[i], precision_list[i], f_list[i])
    print '\n','Correct Rate:',correct
    

if __name__ == '__main__':
    # 训练
    time_start = time.time()
    trainNaiveBayesClassifier()
    print 'Training time:', time.time()-time_start, 's'

    # 测试
    time_start = time.time()
    testNaiveBayesClassifier()
    print 'Testing time:', time.time()-time_start, 's'


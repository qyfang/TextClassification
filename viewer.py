# -*- coding: utf-8 -*-

import os

import sys

import pickle

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.externals import joblib

import matplotlib.pyplot as plt


def _calculate(*matrices):
    labels = os.listdir('data/test/raw/')
    df = pd.DataFrame(columns=('Classifier', 'classification', 'recall', 'precision', 'f1score'))
    index = 0
    for t, confusion_matrix in enumerate(matrices):
        typ = 'Bayes' if t == 0 else 'SVM'
        correct = 0

        r = confusion_matrix.sum(axis=1)
        p = confusion_matrix.sum(axis=0)

        for clf in range(0, 10):
            recall = confusion_matrix[clf][clf] / float(r[clf])
            precision = confusion_matrix[clf][clf] / float(p[clf])
            f1score = 2*recall*precision/(recall+precision)
            df.loc[index] = [typ, labels[clf], recall, precision, f1score]
            index += 1

    return df


def viewMatrix():
    """
    查看数据矩阵
    """
    train_matrix_path = 'matrix/train/matrix.pkl'
    test_matrix_path = 'matrix/test/matrix.pkl'

    train_matrix = joblib.load(train_matrix_path)
    test_matrix = joblib.load(test_matrix_path)

    print 'Train Matrix',train_matrix.shape
    print 'Test Matrix',test_matrix.shape


def viewVocabulary():
    """
    查看数据词典
    """
    reload(sys)
    sys.setdefaultencoding('utf8')

    vocabulary = joblib.load('matrix/vocabulary.pkl')
    for voc in vocabulary.keys():
        print voc,
    

def viewTestResult():
    """
    查看测试结果
    """
    bayes = joblib.load('results/Bayes_confusion_matrix.pkl')
    svm = joblib.load('results/SVM_confusion_matrix.pkl')
    labels = os.listdir('data/test/raw/')

    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    ax = sns.heatmap(bayes, cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    ax.set_title('Bayes')

    plt.subplot(1,2,2)
    ax = sns.heatmap(svm, cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    ax.set_title('SVM')

    plt.subplots_adjust(wspace=0.4, bottom=0.25, top=0.9, right=0.95)
    
    # 绘制每一类的召回率、精确度、F1测度的直方图
    df = _calculate(bayes, svm)
    plt.figure(figsize=(12,8))

    plt.subplot(3,1,1)
    ax = sns.barplot(x='classification',y='recall',hue='Classifier',data=df)
    ax.set_title('Recall')
    ax.set_xlabel('')

    plt.subplot(3,1,2)
    ax = sns.barplot(x='classification',y='precision',hue='Classifier',data=df)
    ax.set_title('Precision')
    ax.set_xlabel('')


    plt.subplot(3,1,3)
    ax = sns.barplot(x='classification',y='f1score',hue='Classifier',data=df)
    ax.set_title('F1score')
    ax.set_xlabel('')

    plt.subplots_adjust(hspace=0.4, bottom=0.07, top=0.96, right=0.93)

    plt.show()



if __name__ == '__main__':
    # viewMatrix()

    # viewVocabulary()

    viewTestResult()

    

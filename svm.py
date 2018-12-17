# -*- coding: utf-8 -*-

import os

import time

import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.svm import LinearSVC, SVC

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from sklearn import metrics


def trainClassifier():
    """
    训练分类器
    """
    # 训练集特征矩阵保存路径
    train_matrix_path = 'matrix/train/matrix.pkl'
    # 分类器保存路径
    classifier_path = 'classifier/classifier.pkl'

    matrix = joblib.load(train_matrix_path)
    target = np.array([x for x in range(10) for i in range(50000)])

    # 构造分类器
    estimators = (
        TfidfTransformer(),
        LinearSVC()
    )
    classifier_params = {
        'tfidftransformer__sublinear_tf': True,
    }
    classifier = make_pipeline(*estimators)
    classifier.set_params(**classifier_params)

    # classifier.fit(matrix, target)
    # best_model = classifier

    parameters = {
        'linearsvc__C': np.arange(0.7, 1.3, 0.1),
        'linearsvc__class_weight': [{0:a, 4:b, 6:c, 7:d} for a in [0.8,1.2,1.6] for b in [0.8,1.2,1.6] for c in [0.8,1.2,1.6] for d in [0.8,1.2,1.6]],
    }
    grid = GridSearchCV(classifier, parameters, cv=5, n_jobs=3)
    grid.fit(matrix, target)
    
    print 'params',grid.best_params_
    print 'score',grid.best_score_

    # 保存分类器
    best_model = grid.best_estimator_
    joblib.dump(best_model, classifier_path)


def testClassifier():
    """
    测试训练集
    """
    # 测试集特征矩阵保存路径
    train_matrix_path = 'matrix/test/matrix.pkl'
    # 分类器保存路径
    classifier_path = 'classifier/classifier.pkl'

    matrix = joblib.load(train_matrix_path)
    target = np.array([x for x in range(10) for i in range(50000)])

    # 读取分类器
    classifier = joblib.load(classifier_path)
    predicted = classifier.predict(matrix)

    term_file_folder_path = 'data/test/raw/'
    confusion_matrix = metrics.confusion_matrix(target, predicted)
    print confusion_matrix
    print metrics.classification_report(target, predicted, target_names=os.listdir(term_file_folder_path))
    print metrics.accuracy_score(target, predicted)

    joblib.dump(confusion_matrix, 'results/SVM_confusion_matrix.pkl')


if __name__ == '__main__':
    # 训练
    # time_start = time.time()
    # trainClassifier()
    # print 'Training time:', time.time()-time_start, 's'

    # 测试
    time_start = time.time()
    testClassifier()
    print 'Testing time:', time.time()-time_start, 's'

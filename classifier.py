# -*- coding: utf-8 -*-

import sys
import os

import time

import pickle

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn import metrics


def readTerm(term_file_folder_path):
    """
    读取Term文件,返回Term字符串生成器和类别生产器
    """
    def getTerm():
        classification = os.listdir(term_file_folder_path)
        for num, clsf in enumerate(classification):
            print num,'/',len(classification)
            for term_filename in os.listdir(term_file_folder_path+clsf):
                path = term_file_folder_path + clsf + '/' + term_filename
                with open(path, 'r') as f:
                    term_list = pickle.load(f)
                term = ' '.join(term_list)
                yield term

    def getTarget():
        classification = os.listdir(term_file_folder_path)
        for num, clsf in enumerate(classification):
            for term_filename in os.listdir(term_file_folder_path+clsf):
                yield num

    #Term字符串生成器
    term_generator = getTerm()
    #Term的类别生成器
    target_generator = getTarget()

    return term_generator, target_generator


def trainClassifier():
    """
    训练分类器
    """
    #训练集的分词数据的文件夹路径
    term_file_folder_path = 'data/train/term/'
    #分类器保存路径
    classifier_path = 'classifier/classifier.pkl'

    #读取数据
    term_generator, target_generator = readTerm(term_file_folder_path)
    target = np.array([x for x in target_generator])

    #构建分类器
    estimators = (
        CountVectorizer(), 
        LatentDirichletAllocation(),
        MultinomialNB(),
        )
    classifier_params = {
        'countvectorizer__min_df': 50,#0.001
        'countvectorizer__max_df': 1.0,
        'latentdirichletallocation__n_components': 5000,
        'latentdirichletallocation__learning_method': 'online',
        'multinomialnb__alpha': 1.0
    }
    classifier = make_pipeline(*estimators)
    classifier.set_params(**classifier_params)

    #训练
    classifier.fit(X=term_generator, y=target)

    #保存分类器
    joblib.dump(classifier, classifier_path)


def testClassifier():
    """
    测试训练集
    """
    #测试集的分词数据的文件夹路径
    term_file_folder_path = 'data/test/term/'
    #分类器保存路径
    classifier_path = 'classifier/classifier.pkl'

    #读取数据
    term_generator, target_generator = readTerm(term_file_folder_path)

    #读取分类器
    classifier = joblib.load(classifier_path)

    #预测
    predicted = classifier.predict(term_generator)
    target = np.array([x for x in target_generator])

    print metrics.classification_report(target, predicted, target_names=os.listdir(term_file_folder_path))
    print metrics.confusion_matrix(target, predicted)
    print metrics.accuracy_score(target, predicted)


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    # #训练
    # time_start = time.time()
    # trainClassifier()
    # print 'Training time:', time.time()-time_start, 's'

    #测试
    time_start = time.time()
    testClassifier()
    print 'Testing time:', time.time()-time_start, 's'

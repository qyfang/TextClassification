# -*- coding: utf-8 -*-

import os

import pickle

import numpy as np

from sklearn.externals import joblib

import matplotlib.pyplot as plt


def viewTerm(classifier, term_file_folder_path='data/train/term/'):
    def getTerm():
        classification = os.listdir(term_file_folder_path)
        for clsf in classification:
            for term_filename in os.listdir(term_file_folder_path+clsf):
                path = term_file_folder_path + clsf + '/' + term_filename
                with open(path, 'r') as f:
                    term_list = pickle.load(f)
                term = ' '.join(term_list)
                yield term

    if 'countvectorizer' in classifier.named_steps:
        term_generator = getTerm()
        countvectorizer = classifier.named_steps['countvectorizer']

        matrix = countvectorizer.transform(term_generator)
        features = (matrix!=0).sum(axis=0)
        feature_num = matrix.shape[1]

        boundary = 50
        hist,bins = np.histogram(a=features, bins=range(1,boundary+2))
        total_h, props = 0, []
        for h in hist:
            total_h += h
            prop = float(total_h)/feature_num
            props.append(prop)

        print 'Total number of features'
        print 'Before filter', feature_num
        print 'After  filter', feature_num-total_h


        plt.bar(bins[:-1], props)
        plt.show()


                


def viewCountVectorizer(classifier):
    if 'countvectorizer' in classifier.named_steps:
        countvectorizer = classifier.named_steps['countvectorizer']
        print countvectorizer.vocabulary_


def viewLatentDirichletAllocation(classifier):
    if 'latentdirichletallocation' in classifier.named_steps:
        lda = classifier.named_steps['latentdirichletallocation']


if __name__ == '__main__':
    #分类器保存路径
    classifier_path = 'classifier/classifier.pkl'

    #读取分类器
    classifier = joblib.load(classifier_path)

    # viewCountVectorizer(classifier)

    # viewTerm(classifier)

    

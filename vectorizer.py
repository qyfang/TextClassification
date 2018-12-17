# -*- coding: utf-8 -*-

import sys
import os

import time

import pickle

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.externals import joblib



def readTerm(term_file_folder_path):
    """
    读取Term文件,返回Term字符串生成器和类别生产器
    """
    def getTerm():
        classification = os.listdir(term_file_folder_path)
        for num, clsf in enumerate(classification):
            print num,'/',len(classification)
            for term_filename in os.listdir(term_file_folder_path+clsf)[:50000]:
                path = term_file_folder_path + clsf + '/' + term_filename
                with open(path, 'r') as f:
                    term_list = pickle.load(f)
                term = ' '.join(term_list)
                yield term

    def getTarget():
        classification = os.listdir(term_file_folder_path)
        for num, clsf in enumerate(classification):
            for term_filename in os.listdir(term_file_folder_path+clsf)[:50000]:
                yield num

    # Term字符串生成器
    term_generator = getTerm()
    # Term的类别生成器
    target_generator = getTarget()

    return term_generator, target_generator


def generateMatrix():
    """
    生成特征矩阵
    """
    vectorizer = CountVectorizer(min_df=0.001)

    for x in ['train', 'test']:
        # 分词数据的文件夹路径
        term_file_folder_path = 'data/%s/term/' % x
        # 特征矩阵保存路径
        matrix_path = 'matrix/%s/matrix.pkl' % x

        # 读取数据
        term_generator, target_generator = readTerm(term_file_folder_path)

        # 训练集拟合后转换为矩阵，测试集根据拟合好的矢量器直接转换为矩阵
        if x == 'train':
            matrix = vectorizer.fit(term_generator)
            # matrix = vectorizer.fit_transform(term_generator)
            joblib.dump(vectorizer.vocabulary_, 'matrix/vocabulary.pkl')

        elif x == 'test':
            matrix = vectorizer.transform(term_generator)

        # 保存特征矩阵
        # joblib.dump(matrix, matrix_path)
        # print matrix.shape
        break



if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    time_start = time.time()
    generateMatrix()
    print 'Transform time:', time.time()-time_start, 's'




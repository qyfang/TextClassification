# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def deduplicateStopWords(filepath):
    deduplicated = []
    i = 0
    with open(filepath, 'rb') as f:
        for line in f.readlines():
            i += 1
            word = line.strip()
            if word not in deduplicated:
                deduplicated.append(word)
            else:
                print i

    return deduplicated

def writeNewStopWordsList(filepath, stopwords):
    with open(filepath, 'w') as f:
        for word in stopwords:
            f.write(word + '\n')

if __name__ == '__main__':
    stopwords = deduplicateStopWords('stopwords/stopwords.txt')
    print 'Completion Detection'
    # writeNewStopWordsList('stopwords.txt', stopwords)
    # print 'Completion Deduplicate'

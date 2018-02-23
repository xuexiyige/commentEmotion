import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pprint as pp


def raw2word(raw_text):
    comment_text = BeautifulSoup(raw_text, 'html.parser').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", comment_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    words_meaningful = [w for w in words if w not in stops]
    return " ".join(words_meaningful)
    pass


if __name__ == '__main__':

    train_data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)[0:100]
    train_data_size = train_data['review'].size

    comments = []

    for i in range(train_data_size):
        #
        if (i + 1) % 1000 == 0:
            print "Review %d of %d" % (i + 1, train_data_size)

        comments.append(raw2word(train_data["review"][i]))
    pp.pprint(comments[0:10])

    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    train_data_features = vectorizer.fit_transform(comments).toarray()
    #
    print train_data_features.shape
    print train_data_features[0][0:10]

    voca = vectorizer.get_feature_names()
    dist = np.sum(train_data_features, axis=0)
    for count, tag in sorted([(count, tag) for tag, count in zip(voca, dist)], reverse=True)[1:20]:
        print(count, tag)

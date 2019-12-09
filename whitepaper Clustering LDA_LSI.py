import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import csv
import nltk
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import pyLDAvis.gensim
from gensim import corpora
from gensim import similarities
from gensim.models import TfidfModel
from gensim.models import LsiModel,LdaModel
from gensim.similarities import MatrixSimilarity
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.corpora import Dictionary

nltk.download('wordnet')
download('punkt')
download('stopwords')

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer




def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)




corpusdir = 'C:/Users/zoha-jooon/Desktop/New folder/txts'  # Directory of corpus.
all_files = PlaintextCorpusReader(corpusdir, '.*')
fileids = all_files.fileids()
print(fileids)
print(len(fileids))
fileindex = []
texts = []
i = 0

for fileid in fileids:
    texts.append(all_files.raw(fileids=fileid))
    fileindex.append(fileid)
    i += 1


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer= PorterStemmer()

def preprocess(text):
    lemmatized_words = []
    text = text.lower()
    doc = word_tokenize(text)
    doc=[re.sub(r'([\w\.-]+@[\w\.-]+\.\w+)', '', word) for word in doc]
    doc=[re.sub(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]| \
            [a-z0-9.\-]+[.][a-z]{2,4}/|[a-z0-9.\-]+[.][a-z])(?:[^\s()<>]+|\(([^\s()<>]+| \
            (\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', word) for word in doc]
    doc= [re.sub(r'https\S+', '', word) for word in doc]
    doc = [re.sub(r'http\S+', '', word) for word in doc]
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    doc = [get_lemma(word) for word in doc]
   # doc = [stemmer.stem(word)for word in doc]

    return doc


texts_og = texts
corpus = [preprocess(text) for text in texts]
number_of_docs = len(corpus)
print(number_of_docs)

def filter_docs(corpus, texts, labels, condition_on_doc):
    number_of_docs = len(corpus)
    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)


corpus, texts, texts_og = filter_docs(corpus, texts, texts_og, lambda doc: (len(doc) != 0))
# pridule implements the concept of Dictionary- a mapping between words and their integer ids, created from corpus.
dictionary = corpora.Dictionary(corpus)
print(dictionary)  # prints unique tokens (words) in corpus
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]  ## converts to B.O.W model- {word, frequency}
# print corpus_gensim
tfidf = TfidfModel(corpus_gensim)
sims = {'texts': {}}

# This moidfModel(corpus_gensim)  # Runs TFID on bag of words model
corpus_tfidf = tfidf[corpus_gensim]

lda_model = LdaModel(corpus_tfidf, id2word=dictionary, num_topics=20, alpha=[0.01]*20,eta=[0.01]*len(dictionary.keys()))
lda_index = MatrixSimilarity(lda_model[corpus_tfidf])

sims['texts']['LDA'] = np.array([lda_index[lda_model[corpus_tfidf[i]]]
                                 for i in range(len(corpus))])

topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)
#
for i,topic in lda_model.show_topics(formatted=True, num_topics=5, num_words=10):
    print(str(i)+": "+ topic)
    print()
# lda_model.save('model5.gensim')
#
# lda = lda_model.load('model5.gensim')


lda_display = pyLDAvis.gensim.prepare(lda_model,corpus_tfidf, dictionary, sort_topics=False)
pyLDAvis.show(lda_display)

lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=93)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sims['texts']['LSI'] = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                 for i in range(len(corpus))])
# %matplotlib inline

linked = linkage(sims['texts']['LSI'], 'complete')

plt.figure(figsize=(10, 20))
dendrogram(linked,
           orientation='left',
           labels=fileindex,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()

linked = linkage(sims['texts']['LDA'], 'complete')

plt.figure(figsize=(10, 20))
dendrogram(linked,
           orientation='left',
           labels=fileindex,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()
data_array = linked

# Initialize figure by creating upper dendrogram
fig = ff.create_dendrogram(data_array, orientation='bottom', labels=fileindex)
for i in range(len(fig['data'])):
    fig['data'][i]['yaxis'] = 'y2'

# Create Side Dendrogram
dendro_side = ff.create_dendrogram(data_array, orientation='right')
for i in range(len(dendro_side['data'])):
    dendro_side['data'][i]['xaxis'] = 'x2'

# Add Side Dendrogram Data to Figure
for data in dendro_side['data']:
    fig.add_trace(data)

# Create Heatmap
dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
dendro_leaves = list(map(int, dendro_leaves))
data_dist = pdist(data_array)
heat_data = squareform(data_dist)
heat_data = heat_data[dendro_leaves,:]
heat_data = heat_data[:,dendro_leaves]

heatmap = [
    go.Heatmap(
        x = dendro_leaves,
        y = dendro_leaves,
        z = heat_data,
        colorscale = 'Blues'
    )
]
print("aval")
heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

# Add Heatmap Data to Figure
for data in heatmap:
    fig.add_trace(data)
print("vasat")
fig.show()
# Edit Layout
fig.update_layout({'width':800, 'height':800,
                         'showlegend':False, 'hovermode': 'closest',
                         })
# Edit xaxis
fig.update_layout(xaxis={'domain': [.15, 1],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  'zeroline': False,
                                  'ticks':""})
# Edit xaxis2
fig.update_layout(xaxis2={'domain': [0, .15],
                                   'mirror': False,
                                   'showgrid': False,
                                   'showline': False,
                                   'zeroline': False,
                                   'showticklabels': False,
                                   'ticks':""})

# Edit yaxis
fig.update_layout(yaxis={'domain': [0, .85],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  'zeroline': False,
                                  'showticklabels': False,
                                  'ticks': ""
                        })
# Edit yaxis2
fig.update_layout(yaxis2={'domain':[.825, .975],
                                   'mirror': False,
                                   'showgrid': False,
                                   'showline': False,
                                   'zeroline': False,
                                   'showticklabels': False,
                                   'ticks':""})
print("1234 payan")

# Plot!
fig.show()


# FIND WAY TO
# def most_similar(i, X_sims, topn=None):
#     """return the indices of the topn most similar documents with document i
#     given the similarity matrix X_sims"""
#
#     r = np.argsort(X_sims[i])[::-1]
#     if r is None:
#         return r
#     else:
#         return r[:topn]
#
#
# # print sims['texts']['LSI']
# results = most_similar(36, sims['texts']['LSI'], 5)
# print("\nMost similar papers to ", fileindex[36], "\n")
# for idx, val in enumerate(results):
#     print(fileindex[val])
# print(most_similar(36, sims['texts']['LSI'], 5))
# Returns the most similar whitepapers to yours
# output = 'C:/Users/amir/Desktop/file.csv'
# corpus.to_csv(output)

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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
download('punkt')
download('stopwords')

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer



############################  Read Data  #############################################

#if you run this code with ipython you should enter the address of your file: instead of './txts'
corpus_directory = './txts'
textsfile = PlaintextCorpusReader(corpus_directory, '.*')
ID_files = textsfile.fileids()
print(ID_files,len(ID_files))



##############################  Preprossesing Data  ######################################

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer= PorterStemmer()
Index_of_files = []
texts = []
count = 0

#file with file_ids
for fileid in ID_files:
    texts.append(textsfile.raw(fileids=fileid))
    Index_of_files.append(fileid)
    count += 1


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return WordNetLemmatizer().lemmatize(word)
    else:
        return WordNetLemmatizer().lemmatize(lemma)
    
def clean_preprocessing(text):
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
    return doc

raw_texts = texts
corpus = [clean_preprocessing(text) for text in texts]
number_of_docs = len(corpus)
print(number_of_docs)


################################   Check our conditions on data and update  #####################################
def check_docs(corpus, texts, labels, state_doc):
    number_of_docs = len(corpus)
    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if state_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if state_doc(doc)]
    corpus = [doc for doc in corpus if state_doc(doc)]
    print("{} docs removed".format(number_of_docs - len(corpus)))
    return (corpus, texts, labels)

# Check some situations and update docs : in this case we delete the empty docs
corpus, texts, raw_texts = check_docs(corpus, texts, raw_texts, lambda doc: (len(doc) != 0))

# implement the concept of Dictionary, words and their integer ids, created from corpus.
dictionary = corpora.Dictionary(corpus)

# now we can see unique tokens (words) in corpus
print(dictionary)

############################################  Create Bag of words and Run TfIdfModel ##################################
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
# Tfidfmodel:The Term Frequency – Inverse Document Frequency(TF-IDF)
tfidf = TfidfModel(corpus_gensim)
#  Runs TFID on bag of words model
corpus_tfidf = tfidf[corpus_gensim]









##############################################   Hirarchial clustering with cosine similarity  #######################
TfidfVec = TfidfVectorizer(stop_words='english')
def cos_similarity(texts):
    tfidf = TfidfVec.fit_transform(texts)
    return (tfidf * tfidf.T).toarray()
cos_similarity(texts)
dist = 1 - cos_similarity(texts)


############################################   Dendrogram Based on distance  for Hirarchial Clustering ################
# you can see the result ===>Figure_1.png

#linkage_matrix with complete role : this object depends on our data. in this case complete and ward are the best
linkage_matrix = linkage(dist, 'complete')

plt.figure(figsize=(10, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('distance')
dendrogram(linkage_matrix,
           orientation='left',
           labels=Index_of_files,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()







#########################################################   Run LDA Model  ###########################################
#for LDA model we can set 2 more parameters alpha and beta like this alpha=[0.01]*20,eta=[0.01]*len(dictionary.keys()  but in real projects, we dont need these
#create similarity matrix
sims = {'texts': {}}
lda_model = LdaModel(corpus_tfidf, id2word=dictionary, num_topics=20)
lda_index = MatrixSimilarity(lda_model[corpus_tfidf])
sims['texts']['LDA'] = np.array([lda_index[lda_model[corpus_tfidf[i]]]
                                 for i in range(len(corpus))])
# we can see all 20 topics : this algorithm create each topic based on 10 word then put all related words on this topic
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)
# we can see 5 topics and thier words
for i,topic in lda_model.show_topics(formatted=True, num_topics=5, num_words=10):
    print(str(i)+": "+ topic)
    print()
    
    
###################################3############     visualization of Lda Algorithm   ################################
#you should run these 2 lines in ipython 
#lda_display = pyLDAvis.gensim.prepare(lda_model,corpus_tfidf, dictionary, sort_topics=False)
#pyLDAvis.show(lda_display)


#################################################   visualization of Lda Algorithm with dendrogram ####################
# you can see the result ===>Figure_2.png
# if you want to see the accurate result of Lda you should see

linked = linkage(sims['texts']['LDA'], 'complete')

plt.figure(figsize=(10, 20))
plt.title('LDa Clustering Dendrogram')
dendrogram(linked,
           orientation='left',
           labels=Index_of_files,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()








#################################################      Run Lsi Model      ############################################
#you can change number of topics(num_topics=20) and see diffrent results

lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sims['texts']['LSI'] = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                 for i in range(len(corpus))])


#################################################   visualization of Lsi Algorithm with dendrgram #####################
# you can see the result ===>Figure_3.png
linked = linkage(sims['texts']['LSI'], 'complete')


plt.figure(figsize=(10, 20))
plt.title('Lsi Clustering Dendrogram')
dendrogram(linked,
           orientation='left',
           labels=Index_of_files,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()







###########################       Compare LSi and Hirarchial results with Heatmap and dendrogram     ###################


Lsi_linked = linked

# Initialize figure by creating Lsi Dendrogram (upper)
fig = ff.create_dendrogram(Lsi_linked, orientation='bottom', labels=Index_of_files)
for i in range(len(fig['data'])):
    fig['data'][i]['yaxis'] = 'y2'

# Create Hirarchial Dendrogram (side)
Hirarchial_linked=linkage_matrix
dendro_side = ff.create_dendrogram(Hirarchial_linked, orientation='right')
for i in range(len(dendro_side['data'])):
    dendro_side['data'][i]['xaxis'] = 'x2'

# Add Side Dendrogram Data to Figure
for data in dendro_side['data']:
    fig.add_trace(data)

# Create Heatmap
dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
dendro_leaves = list(map(int, dendro_leaves))
data_dist = pdist(Lsi_linked)
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
heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

# Add Heatmap Data to Figure
for data in heatmap:
    fig.add_trace(data)
#fig.show()
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





##############################################     Most similar function using Lsi  ########################################
"" "Find top K most similar texts with text[i] for instance( Aion-AION-Whitepaper.txt )"""
K=5
i=1
def K_Similar_to_one(i, K_similar, topn=None):
    
    r = np.argsort(K_similar[i])[::-1]
    if r is None:
        return r
    else:
        return r[:topn]

results = K_Similar_to_one(i, sims['texts']['LSI'], K)
print("\nMost similar texts to ", Index_of_files[i], "\n")
for idx, val in enumerate(results):
    print(Index_of_files[val])
#print(K_Similar_to_one(i, sims['texts']['LSI'], K))
# you should detemine a path to save this data
output = 'C:/Users/zoha-jooon/Desktop/file.csv'

# clustering

import all packages ...

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

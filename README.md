# What is Text Analysis?
Text analysis allows companies to automatically extract and classify information from text, such as tweets, emails, support tickets, product reviews, and survey responses. Popular text analysis techniques include sentiment analysis, topic detection, and keyword extraction.

# Whitepaper Clustering

In this project, the goal is to cluster 94 whitepapers related to Etherium. Each whitepaper is a big text file that consists of almost 3000 lines.
To achieve this aim, I have implemented some Algorithms, then compared their results.
In this case, I have 94 files. Each file contains many text data. So, my work consists of 5 steps.
### 1-Reading the files
### 2-Preprocessing
### 3-Tokenizing
### 4-Creating Bag of words and Run Tf_Idf Model
### 5-Run clustering Algorithms and Show their results 
## 
##
## Clustering
we can cluster these data with two approaches:
### 1-semantic clustering
### 2-lexical clustering
For instance, how similar are the phrases “the cat ate the mouse” with “the mouse ate the cat food” by just looking at the words?
On the surface, if you consider only word-level similarity, these two phrases appear very similar to 3 of  4 unique words are an exact overlap. It typically does not take into account the actual meaning behind words or the entire phrase in context.
Instead of doing a word for word comparison, we also need to pay attention to the context in order to capture more of the semantics.
To consider semantic similarity, we need to focus on phrase/paragraph levels (or lexical chain level) where a piece of text is broken into a relevant group of related words before computing similarity.We know that while the words significantly overlap, these two phrases actually have a different meaning

## Read data 
for reading files, we need to run this part of code

############################  Read Data  #############################################

#if you run this code with ipython you should enter the address of your file: instead of './txts'
corpus_directory = './txts'
textsfile = PlaintextCorpusReader(corpus_directory, '.*')
ID_files = textsfile.fileids()
print(ID_files,len(ID_files))

## Preprocessing
Raw data contain numerical values, punctuation, special character, etc. These values can hamper the performance of the model, so before applying any text featurization. First, we need to convert raw data into meaningful data, which is also called as text preprocessing. This step can be done by following ways:
#### 1-convert all characters to lower case
#### 2-remove all stopwords, punctuations, numbers, HTML tags
#### 3-Tokenization 
### -Remove Noisy Data
In regular sentences, Noisy data can be defined as text file header, footer, HTML, XML, markup data. As, this type of data is not meaningful and does not provide any information, so it is mandatory to remove this type of noisy data. In python HTML, XML can be removed by BeautifulSoup library while markup, the header can be removed by using regular expression
### -Tokenization
In tokenization, we convert the group of sentences into token. It is also called text segmentation or lexical analysis. It is basically splitting data into a small chunk of words. For example- We have sentence — “Ross 128 is an earth-like planet. Can we survive on that planet?”. After tokenization, this sentence will become -[‘Ross’, ‘128’, ‘is,’ ‘earth,’ ‘like,’ ‘planet,.’ ‘’, ‘Can,’ ‘we,’ ‘survive’, ‘in’, ‘that’, ‘planet’, ‘?’]. Tokenization in python can be done by python’s NLTK library’s word_tokenize() function.
To remove redundant words that are from the same root as ability and abilities. we can lemmatized words with the 
WordNetLemmatizer function.
## Check function
Now we have file_ids and their data, however in the progress of preprocessing its possible for a file to get empty, so we must check that if any file is empty after preprocessing, we should remove its ID. I have implement the check function. So , the result was interesting two documents out of 94  Documents removed.
### Note
For running next parts of the code faster, we can save the preprocessed data in this step
the next step is to find the frequency of words and find Tf-Idf matrix. 

## TF-IDF
TF-IDF stands for Term Frequency-Inverse Document Frequency, which basically tells the importance of the word in the corpus or dataset. TF-IDF contain two concept Term Frequency(TF) and Inverse Document Frequency(IDF).

## Term Frequency
Term Frequency is defined as how frequently the word appears in the document or corpus. As each sentence is not the same length so it may be possible a word appears in long sentences occur more time as compared to word appear in sorter sentence.

## Cosine_similarity Function 
the process of converting text into a vector is called vectorization.
By using the CountVectorizer function, we can convert a text document to the matrix of word count.
After applying the CountVectorizer, we can map each word to feature indices. We use these vectors to calculate cosine similarity.
### Definiation of Cosine similarity
Cosine similarityis not only telling the similarity between two vectors, but it also tests the orthogonality of vector. Where theta is the angle between two vectors, if the angle is close to zero, then we can say that vectors are very similar to each other.

If theta is 90, then we can say vectors are orthogonal to each other (orthogonal vector not related to each other ), and if theta is 180, we can say that both the vector are opposite to each other.

## Hierarchical clustering
Now everything is ready to use a hierarchical clustering algorithm with the use of cosine similarity matrix
then we can use results in Linkage function in, complete, ward or simple method in order to draw dendrogram.
Ward method is actually a method that tries to minimize the variance within each cluster. 
as you can see in Figure-1, the proper number of clusters for this data is about 5-8 clusters
                                click on image to see in detail
<a href="https://github.com/zohreh-aa/clustering/blob/master/Figure_1.png"><img src="https://github.com/zohreh-aa/clustering/blob/master/Figure_1.png" title="Hierarchical clustering" alt="Hierarchical clustering"></a>

## LSI or LSA Model
Latent Semantic Analysis, or LSA, is one of the foundational techniques in topic modeling. The core idea is to take a matrix of what we have — documents and terms — and decompose it into a separate document-topic matrix and a topic-term matrix.
The first step is generating our document-term matrix. Given m documents and n-words in our vocabulary, we can construct an m × n matrix A in which each row represents a document, and each column represents a word. In the simplest version of LSA, each entry can be a raw count of the number of times the jth word appeared in the ith document. In practice, however, raw counts do not work particularly well because they do not account for the significance of each word in the document. For example, the word “nuclear” probably informs us more about the topic(s) of a given document than the word “test.”
Consequently, LSA models typically replace raw counts in the document-term matrix with a tf-idf score.
You can see the result of this model in Figure-2.

                                click on image to see in detail
<a href="https://github.com/zohreh-aa/clustering/blob/master/Figure_2.png"><img src="https://github.com/zohreh-aa/clustering/blob/master/Figure_2.png" title="LSA" alt="LSA"></a>

## LDA Model
To get a better result, we run an excellent algorithm for unsupervised data clustering known as the LDA algorithm (the abbreviation LDA stands for Latent Dirichlet allocation). With the assumption of 20 topics, we could see that some of the clusters have overlap. This part has an interactive diagram so that you can see the most critical words in each topic. This overlap means that the meaning of overlapped topic is almost the same.
Because of this close meaning in some topics, it is suggested not to use word embedding algorithms.
LDA stands for Latent Dirichlet Allocation. LDA is a Bayesian version of pLSA. In particular, it uses dirichlet priors for the document-topic and word-topic distributions, lending itself to better generalization. you can see the result on LDa output.rar and Figure-3.
### NOTE
we found 16000 unique words in these files that all raw words was 22500 in all.
Then we run LSI algorithms, and we compared the result of these algorithms with the previous one using a heatmap, you can see Fig-4 :
Then with LSI, we implement a function to find the most similar white paper to one specific whitepaper; for example, if you want the most similar paper to paper number 2, you can use this function. The result is :


## Final words

This report was written mainly to present an idea about unsupervised language processing, not to create the best possible solution based on it, so there is plenty of space to improve it. Improvements that come into my mind include:
#### 1-K-Means clustering based on cosine,
#### 2-Removing Chinese Stop Words 
#### 3-Implement CBOW on our data.
When we looked at the words of each cluster, we found that some words like Chinese characters had some negative effect on clustering, which we can improve the results by removing them.
Word2Vec is a method to construct such an embedding. It can be obtained using two methods (both involving Neural Networks):
### 1-Skip Gram
### 2-Common Bag Of Words (CBOW)
Both have their advantages and disadvantages. According to researchers, Skip Gram works well with a small amount of data and is found to represent rare words well. On the other hand, CBOW is faster and has better representations for more frequent words.
So, we can implement CBOW on our data.
Here we arrive at the end of this report.

#### Thank you for reading it.
To view the code and execute the written code you can see RUn and result.rar.


## Refrences

###### 1-(https://www.kernix.com/article/similarity-measure-of-textual-documents/)
###### 2-(https://medium.com/@adriensieg/text-similarities-da019229c894)
###### 3-(https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)
###### 4-(https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483)
###### 5-(https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318)
###### 6-(https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)
###### 7-(https://medium.com/towards-artificial-intelligence/text-mining-in-python-steps-and-examples-78b3f8fd913b)
###### 8-(https://www.learntek.org/blog/categorizing-pos-tagging-nltk-python/)
###### 9-(https://www.nltk.org/)
###### 10-(https://expertsystem.com/natural-language-processing-and-text-mining/)

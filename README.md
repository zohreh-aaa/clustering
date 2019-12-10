# What is Text Analysis?
Text analysis allows companies to automatically extract and classify information from text, such as tweets, emails, support tickets, product reviews, and survey responses. Popular text analysis techniques include sentiment analysis, topic detection, and keyword extraction.

# Whitepaper Clustering

In this project, the goal is to cluster 94 whitepapers related to Etherium. Each whitepaper is a big text file that consists of almost 3000 lines.
To achieve this aim, I have implemented some Algorithms, then compared their results.
In this case, I have 94 files. Each file contains many text data. So, my work consists of 5 steps.
## 1-Reading the files
## 2-Preprocessing
## 3-Tokenizing
## 4-Creating Bag of words and Run Tf_Idf Model
## 5-Run clustering Algorithms and Show their results 

# about clustering
we can cluster these data with two approaches 
## semantic clustering
## lexical clustering
For instance, how similar are the phrases “the cat ate the mouse” with “the mouse ate the cat food” by just looking at the words?


. On the surface, if you consider only word-level similarity, these two phrases appear very similar to 3 of the 4 unique words are an exact overlap. It typically does not take into account the actual meaning behind words or the entire phrase in context.


. Instead of doing a word for word comparison, we also need to pay attention to the context in order to capture more of the semantics. To consider semantic similarity, we need to focus on phrase/paragraph levels (or lexical chain level) where a piece of text is broken into a relevant group of related words prior to computing similarity. We know that while the words significantly overlap, these two phrases actually have a different meaning.
## read data 
for reading files, we need to run this part of code


## preprocessing
before we could run any algorithms on data, we should preprocess data, 
we should :
### convert all characters to lower case
### remove all stopwords, punctuations, numbers, HTML tags 

## tokenizing
To remove redundant words that are from the same root like ability and abilities.
we can tokenize words
with this part of code

know we have file_ids and their data, however in the progress of preprocessing its possible for a file to get empty, so we must check that if any file is empty after preprocessing, we should remove its ID
we do so by this part of code:







for running next parts of the code faster, we can save the preprocessed data in this step

## bag of word and tfidf
the next step is to find the frequency of words and find Tf-Idf matrix 



## now everything is ready to use hierarchical clustering algorithm
we the use of cosine similarity matrix 

then we can use results in Linkage function in, complete, ward or simple type in order to draw dendrogram
the output is :




as you can see the proper number of clusters for this data is about 5-8 clusters 

In order to get a better result, we run an excellent algorithm for supervised data clustering known as LDA algorithm (the abbreviation LDA stands for Latent Dirichlet allocation)
With the assumption of 20 topics, we could see that some of the clusters have overlap. This part has an interactive diagram so that you can see the most important words in each topic.
This overlap means that the meaning of overlapped topic is almost the same.

because of this close meaning in some topics, it is suggested not to use word embedding algorithms 

When we looked at topic words of clusters, we found that some words like Chinese words had some negative effect on clustering, which we can improve the results by removing them.
we found 16000 unique words in this files that all raw words was 22500 in all 

then we run LSI algorithms, and we compared the result of these algorithms with the previous one by means of a heatmap 
you can see file 4 :





Then with LSI, we implement a function to find the most similar white paper to one specific whitepaper; for example, if you want the most similar paper to paper number 36, you can use this function.
The result is :

## final words
to wrap it up, we can mention that algorithms like k-means or KNN are not good at this particular case, so we skip them.

## Refrences

1-(https://www.kernix.com/article/similarity-measure-of-textual-documents/)
2-

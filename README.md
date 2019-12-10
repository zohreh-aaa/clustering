# What is Text Analysis?
Text analysis allows companies to automatically extract and classify information from text, such as tweets, emails, support tickets, product reviews, and survey responses. Popular text analysis techniques include sentiment analysis, topic detection, and keyword extraction.

# Whitepaper Clustering

In this project,the goal is to cluster 94 whitepapres related to Etherium, each whitepaper is a big text file cosist of almost 3000 lines.
to achive to this aim I have implemented some Algorithms then compared thier results.
In this case, I have 94 files. Each files contains alot of text's data. So, my work consists of 5 steps.
## 1-Reading the files
## 2-Preprosessing
## 3-Tokenizing
## 4-Creating Bag of words and Run Tf_Idf Model
## 5-Run clustering Algorithms and Show their results 

# about clustering
we can cluster these data with to approaches 
## semantic clustering
## lexical clustering
For instance, how similar are the phrases “the cat ate the mouse” with “the mouse ate the cat food” by just looking at the words?


. On the surface, if you consider only word level similarity, these two phrases appear very similar as 3 of the 4 unique words are an exact overlap. It typically does not take into account the actual meaning behind words or the entire phrase in context.


. Instead of doing a word for word comparison, we also need to pay attention to context in order to capture more of the semantics. To consider semantic similarity we need to focus on phrase/paragraph levels (or lexical chain level) where a piece of text is broken into a relevant group of related words prior to computing similarity. We know that while the words significantly overlap, these two phrases actually have different meaning.
## read data 
for reading files we need to run this part of code


## preprosessing
befor we could run any algorithms on data we should preprosess data, 
we should :
### convert all charecters to lower case
### remove all stopwords, punctuations, numbers, html tags 

## tokenizing
to remove redundent words that are from same root like ability and abilities.
we can tokenize words
with this part of code

know we have file_ids and their data, however in the progress of preprosessing its possible for a file to get empty, so we must check that if any file is empty after preprosessing we should remove it's ID
we do so by this part of code:







for running next parts of the code faster we can save the preprosessed data in this step

## bag of word and tfidf
the next step is to find the frequncy of words and find Tf-Idf matrix 



## now everything is ready to use hirarchial clustering algorithm
we the use of cosine similarity matrix 

then we can use results in Linkage function in complete, ward or simple type in order to draw dendrogram
the output is :




as you can see the proper number of clusters for this data is about 5-8 clusters 

in order to get a better result we run a great algorithm for supervised data known as LDA algothim the abbriviation LDA stands for Latent Dirichlet allocation
with the assumption of 20 topics we could see that some of clusters have overlap this part has an intractive diagram and you can see most important words in each topic.
this overpal means that the meaning of overlaped topic is almost the same.

because of this close meaning in some topicts its sugested not to use word embedding algorithms 

when we looked at topic words of clusters we found that some words like chinese words had some negative effect on clustering which we can improve the results by removing them.
we found 16000 unique words in this files that all raw words was 22500 in all 

then we run LSI algorithms and we compared the result of this algorithms with the previous one by the means of a heatmap 
you can see file 4 :





then with lsi we implement a function to find the most similar white paper to one specific whitepaper for example if you want the most similar paper to paper number 36 you can use this function.
the result is :


to wrap it up we can mention that algorithms like k-means or knn are not good at this particular case so we skip them 
در پایان میتوان عنوان کرد که الگوریتم های کامینز و غیره به علت کارایی \پایین انتخاب نشد 
در نهایت کلاستر بندی داده ها با استفاده از روش های ورد امبدینگ مربوطه میتواند یکی از بهترین راه حل ها برای این مساله باشد یعنی ساختن یک دیتابیس مانند کلمات فیسبوک که فقط شامل داده های متنی بلاک چین و ارز های مجازی باشد میتواند برای 
کار های آینده استفاده شود.
همچنین روش های آماری برای نرمال کردن داده ها نیز می تواند باعث بهبود در نتیجه شود.

## Refrences

1-(https://www.kernix.com/article/similarity-measure-of-textual-documents/)
2-

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

در این پروژه می خواهیم داده های وایت پیپر را کلاستر بندی کنیم. برای این کار دو رویکرد وجود دارد کلاستر بندی مفهومی یا لغتی در کلاستر بندی مفهومی دو متن که از نظر محتوا بهم شبیه هستند در یک خوشه قرار میگیرند ولی در کلاستر بندی لغتی متن ها از نظر تعداد کلمات شبیه به هم در یک خوشه قرار می گیرند مثل مورچه گل را خورد یا گل مورچه را خورد این دو متن از لحاظ کلاستر بندی لغتی شبیه به هم اند اما از لحاظ کلاستر بندی مفهومی کاملا باهم متفاوت اند در این پروژه هدف پیاده سازی الگوریتم های کلاستر بندی مفهومی می باشد. داده های داده شده متنی هستند پس قبل از اینکه الگوریتمی روی این داده ها اجرا کنیم نیاز داریم تا پردازش اولیه انجام دهیم برای این کار ابتدا تمام حروف را به لوورکیس  تبدیل می کنیم سپس باید اعداد ،استاپ ورد ها و پانکچویشن ها و تگ های اچ تی ام ال را حذف کنیم بعد از این کار داده را توکنایز کرده و ریشه کلمات را به دست می آوریم برای مثال کلمه ابلیتی و ابلیتیز هردو از یک ریشه اند و برای فهمیدن مفهوم متن نیاز داریم که ریشه کلمات را ذخیره کنیم 
در انتهای این بخش تغییرات را ذخیره میکنیم در بخش بعدی چک میکنیم که اگر بعد از تغییرات اعمال شده در بخش قبل اگر محتوی فایلی خالی شده  بود آن فایل را از مجموعه داده ها حذف میکنیم. بعد از اجرای این دو بخش برروی داده های خود مشاهده کردیم که دو فایل از بین رفتند یعنی محتویات این دوفایل حاوی کلمات با معنا نبودند 
بعد از این مرحله بگ آو ورد را برای هر لغت بدست آوردیم و مدل تی اف آی دی اف را اجرا کردیم.
سپس با استفاده از تابع مشابهت کوسینوس cosine similarity
الگوریتم خوشه بندی سلسه مراتبی را اجرا کردیم برای اجرای این الگوریتم به  ماتریس مشابهت نیاز داریم این ماتریس میزان شباهت بردار های مربوط به هر متن را از طریق ضرب داخلی دو بردار (کسینوس) محاسبه میکند و مقادیر بدست آمده را در ماتریس ذخیره می کند. 
سپس دندروگرام مربوطه را با Linkage 'complete'
رسم کردیم
با توجه به این مساله و خروجی های تولید شده در پایان به این نتیجه رسیدم که لینکیج کامپلیت و وارد بهترین گزینه ها برای این داده هستند.
در نهایت خروجی این قسمت در فایل شماره یک موجود است
دندروگرام رسم شده میزان کلاستر مناسب را بین ۵ تا ۸ کلاستر نمایش میدهد.
برای مقایسه بهتر یکی از بهترین الگوریتم های تشخیص متن آن سوپروایزد به اسم ال دی ای را روی این دیتا اجرا کردیم.
نتیجه بدست آمده بسیار نزدیک بود میزان کلاستر های مناسب با استفاده از این روش مقدار ۵ تا۶ بدست آمد ونکته مهم این بود که در این روش ما تعداد موضوعات را به طور پیشفرض ۲۰ موضوع انتخاب کردیم اما نتیجه کار به این صورت شد که الگوریتم ۴الی ۵  موضوع را به طور کاملا متفاوت نشان میداد و بقیه موضوعات کاملا با هم ه\وشانی داشتند این موضوع نشان می دهد که دیتا اولیه ما از لحاظ موضوع بسیار مرتبط می باشد  به همین علت روش های ورد امبدینگ به خوبی نمیتواند روی این دیتا پاسخگو باشد همچنین در این مرحله ما بعضی از تاپیک ها به همراه کلمات داخل آن ها را نمایش دادیم که متوجه یک نکته مهم دیگر درون دیتا شدیم. این داده ها حروف مختلفی از زبان های انگیلیسی و چینی را دارند و کلمات چینی نیز در این فایل ها یافت می شود پس یکی از روش هایی که باعث بهبود در نتایج ما خواهد شد استفاده از روش های مربوط به داده های ترکیبی می باشد به عنوان مثال پاک کردن حروف چینی یا ریشه گیری از آن ها و یا حذف اعداد چینی این یکی از روش هاییست که به ما کمک می کند تا داده های نرمال شده و نتایج قابل اعتماد تری را تولید کنیم.
به عنوان مثال در این متن ما توانستیم ۱۶۰۰۰ کلمه یونیک پیدا کنیم در حالیکه که کلمات اولیه متن قبل از پردازش حدود ۲۲۵۰۰۰ بود
بعد از اجرای این الگوریتم الگوریتم ال اس ای را اجرا کردیم و جواب حاصله از این الگوریتم را با الگوریتم هایررکیال مقایسه کردیم این مقایسه را از طریق دندروگرام و هیت مپ انجام دادیم
شکل ۴ در فایل بالا نتیجه را نمایش می دهد.
سپس با استفاده از الگوریتم ال اس آی
کا تا از شبیه ترین متن ها به متن آي ام مورد نظر خود را جستجو میکنیم این روش برای مواقعی که یک متن وایت پیپر داریم و میخواهیم شبیه ترن ها به این متن را پیدا کنیم بسیار ناسب می باشد همچنین برای پیدا کردن تقلب یا داده های تکراری و یا تشخیص میزان کپی برداری نیز میتوان از این روش استفاده کرد در این جا ما متن دوم را انتخاب کردیم و نتیجه به شکل زیر شد
:

در پایان میتوان عنوان کرد که الگوریتم های کامینز و غیره به علت کارایی \پایین انتخاب نشد 
در نهایت کلاستر بندی داده ها با استفاده از روش های ورد امبدینگ مربوطه میتواند یکی از بهترین راه حل ها برای این مساله باشد یعنی ساختن یک دیتابیس مانند کلمات فیسبوک که فقط شامل داده های متنی بلاک چین و ارز های مجازی باشد میتواند برای 
کار های آینده استفاده شود.
همچنین روش های آماری برای نرمال کردن داده ها نیز می تواند باعث بهبود در نتیجه شود.

## Refrences

1-(https://www.kernix.com/article/similarity-measure-of-textual-documents/)
2-

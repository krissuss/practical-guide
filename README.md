# A Practical Guide for Social Media Research using RoBERTa for Sentiment Analysis

### Introduction

Large language models are becoming the status quo for conducting state-of-the-art research, particularly for natural language processing tasks.

In this tutorial, we will employ a RoBERTa model for conducting **Sentiment Analysis** using a dataset of tweets that include publicly available mentions of #HappyBirthdayTaylorSwift, occurring on December 13, 2023.

This guide was created for use with **A practical guide for conducting state-of-the-art social media research** (Sussman, Looi, & Park, 2023) as part of the Journal of Current Issues in Advertising Research — Special Issue “Emerging Issues in Computational Advertising”.

#### Contents of the notebook

The notebook is divided into the following sections to provide researchers with an organized process for employing the RoBERTa model for sentiment analysis. Naturally, the process should be modified for individual research use cases. The sections are:

1. [Importing Python Libraries and preparing the environment](#section01)
2. [Get the data set URL from Kaggle](#section02)
3. [Get Kaggle API token](#section03)
4. [Load the data](#section04)
5. [Sentiment Analysis](#section05)

#### Technical Details

The script used in this guide relies on multiple tools, listed below. Researchers must enable these elements to successfully implement this script.

 - Data:
	 - We are using the csv dataset available at [#HappyBirthdayTaylorSwift Tweet Data](https://kaggle.com/datasets/1d11f844a930184c6b7d2b004c70810797017d7d5c1a72d135a854b90439237d)

 - LLM used:
	 - The RoBERTa model was proposed in RoBERTa: A Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. It is based on Google’s BERT model released in 2018.

- References:
    - [Twitter-roBERTa-base for Sentiment Analysis](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
    - [Paper](https://arxiv.org/pdf/2010.12421.pdf)

- Hardware Requirements:
	 - Python 3.6 and above
	 - Pytorch, Transformers and All the stock Python ML Libraries
	 - GPU enabled setup

 ### Step 1: Import Libraries & Upload the CSV file

Here we import the required libraries; we just need a few to download and view the data sets along with opendatasets

```
!pip install opendatasets

import pandas as pd
pd.set_option('display.max_columns', None)
import os
import opendatasets as od
```

### Step 2: Get the data set URL from Kaggle

Next, we get the Kaggle URL for the specific data set we need to download.

- We are using the csv dataset available at [#HappyBirthdayTaylorSwift Tweet Data](https://kaggle.com/datasets/1d11f844a930184c6b7d2b004c70810797017d7d5c1a72d135a854b90439237d)


### Step 3: Get Kaggle API token

Before we start downloading the data set, we need the Kaggle API token. To get that:
- Login into your Kaggle account
- Get into your account settings page
- At the top right of the screen, click on the elipses / three dots. Then, click on the “Copy API command” button.
- This will prompt you to download the .json file into your system. Save the file, and we will use it in the next step.


### Step 4: Load the data

Now that we have all the required information let’s start downloading the data sets from Kaggle using the steps below. You will need to add your Kaggle credentials: username and key. You can find those by opening up the json file.

"""
```
# Assign the Kaggle data set URL into variable
dataset = 'https://www.kaggle.com/datasets/kristensussman/happybirthdaytaylorswift-data-121323'
# Using opendatasets let's download the data sets
od.download(dataset)
```
Once the file has finished uploading, you can use the `pd.read_csv()` command to read the file into a pandas data frame.

```
# Read and load the data
path = '/content/happybirthdaytaylorswift-data-121323/Master_text_only_122823.csv'
df = pd.read_csv(
    path, index_col=0
)
```

We can use the df.head(10) command to display the first 10 rows of the data frame. This gives us a glimpse of the data and helps us understand the structure of the dataset. The output will show us the column names and the first few rows of the dataset.
```
df.head(10)
```
```
!pip install transformers
!pip install nltk
```
```
import nltk
```

### Step 5: Sentiment Analysis

```
nltk.download()
```
```
import re
import torch
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
from nltk.tokenize import TweetTokenizer

# Load your dataframe (assuming it's already loaded as 'df')

# Load Twitter-RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Tokenizer for tweet-specific preprocessing
tweet_tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))

# Function for tweet preprocessing
def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove mentions and hashtags
    tweet = re.sub(r'\@\w+|\#\w+', '', tweet)
    # Tokenize tweet
    tokens = tweet_tokenizer.tokenize(tweet)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Reconstruct tweet
    preprocessed_tweet = ' '.join(tokens)
    return preprocessed_tweet

# Function to analyze sentiment and assign polarity scores
def analyze_sentiment(tweet):
    preprocessed_tweet = preprocess_tweet(tweet)
    inputs = tokenizer.encode_plus(preprocessed_tweet, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    # Calculate polarity score (positive - negative)
    polarity_score = probabilities[2] - probabilities[0]
    return polarity_score

# Apply sentiment analysis to each tweet in the dataframe
df['Polarity Score'] = df['Full Text'].apply(analyze_sentiment)

# Function to assign sentiment labels based on the polarity score
def assign_sentiment_label(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Create a new column and assign sentiment labels based on the polarity score
df['Sentiment Label'] = df['Polarity Score'].apply(assign_sentiment_label)
```
This code uses the Twitter-RoBERTa model for sentiment analysis. The preprocess_tweet function is used to perform standard tweet-specific preprocessing, including converting text to lowercase, removing URLs, mentions, hashtags, stopwords, and tokenizing the tweet using NLTK's TweetTokenizer.

The analyze_sentiment function preprocesses the tweet using the defined preprocessing steps, encodes it using the Twitter-RoBERTa tokenizer, passes it through the model, computes the probability scores for positive, negative, and neutral sentiments, and calculates the polarity score by subtracting the negative probability from the positive probability.

The assign_sentiment_label function assigns sentiment labels ('Positive', 'Negative', 'Neutral') based on the polarity score, and a new column 'Sentiment Label' in the dataframe is created to store these labels.

Note, the thresholds for determining positive, negative, or neutral sentiment labels (0.1 and -0.1) are arbitrary and can be adjusted based on your specific use case and dataset characteristics.
```
df.head(100)
```
```
# prompt: create a new csv of the df

df.to_csv('df_sentiments.csv', index=False)
```

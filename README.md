# Parallel Clustering based Topic Modeling

## 1. Project Background
  - Topic modeling is an unsupervised learning process that automatically extracts topics or issues from a document set by analyzing the patterns of words constituting text data.
  - Cluster-based topic modeling is a methodology that combines embedding from language models with a clustering architecture.
  - In this project, we create a topic modeling model using parallel clustering and semantic embedding.

## 2. Dataset
  - The Korean National Assembly provided News articles, Twitter, and online community data related to major legislation in Korea.
  - Korean NLI and STS dataset were additionally collected for fine-tuning the language model.

## 3. Overall pipeline
To summarize the entire process of the 『Parallel Clustering based News article Topic Modeling』 we designed, it consists of the following four steps.

  - STEP 1) Unsupervised training(TSDAE) : The language model understands the context of a given news article and optimizes it for the domain through unsupervised learning of the TSDAE method.
 
  - STEP 2) Supervised trainig(NLI and STS) : Our language model trains on the Korean NLI·STS dataset so that the model can distinguish similarities between sentences or documents.
  
  - STEP 3) Parallel Clustering : This clustering method we designed focuses on speed and stability.
  
  - STEP 4) Keyword Extraction : Extracts important words from clustered groups using C-TF-IDF (Class-based Term Freq-Inverse Doc Freq) calculation method.
  
  ![my_lm](https://user-images.githubusercontent.com/105137667/195859373-eeebeba5-c657-4613-96f2-08b8d7479faa.jpg)


### STEP 1) Unsupervised training(TSDAE)
  
  - TSDAE consists of the following three steps:
    1. TSDAE introduces noise to input sequences by deleting or swapping tokens.
    2. These damaged sentences are encoded by the transformer model into sentence embedding.
    3. Another decoder network then attempts to reconstruct the original input from the damaged sentence encoding.

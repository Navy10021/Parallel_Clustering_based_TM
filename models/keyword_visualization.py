########################################
## My Keywords visualization function ##
########################################

import warnings
warnings.filterwarnings(action='ignore') 
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns
from wordcloud import WordCloud
FONT_PATH = 'data/font/NanumBarunGothicBold.ttf'

def get_n_word(n_words):
    """
    n_words : top n words / {word : score}(dict)
    return : preprocessed top n words
    """
    # 1. get keys
    w_list = list(n_words.keys())
    # 2. word list with punct & number filter
    w_list = [word for word in w_list if word.isalpha()]
    # 3. Stop words
    #w_list = [word for word in w_list if word not in stop_words]
    # 4. retun dict
    w_dict = {k : n_words[k] for k in n_words.keys() & set(w_list)}
    
    return w_dict

def word_cloud(n_word_list):
    n_words = dict(n_word_list)
    n_words = get_n_word(n_words)
    
    wc = WordCloud(font_path= FONT_PATH, width = 600, height = 600, background_color='black', collocations=False, min_font_size=10)
    cloud = wc.generate_from_frequencies(n_words)
    plt.figure(figsize=(6, 8))
    plt.axis('off')
    plt.imshow(cloud)
    plt.show()

def news_keywords(topic_name, top_n_words, top_k):
    print(" \n ===== '{}' 관련 기사 토픽 및 키워드 Top {} ===== \n ".format(topic_name, top_k))
    print(">> Total number of topics :", len(top_n_words))
    for i, (key, value) in enumerate(top_n_words.items()):
        word_cloud(value)
        if i > top_k:
            break
import pandas as pd
from preprocessing.text_preprocessing import *
#import pickle5 as pickle

# 1. Make Clean Korean News article DataFrame

# 1-1. Lease 3 Law
col_name = ['Date', 'Paper', 'Review', 'Title', "Text"]
data_path_1 = './data/임대차3법/임대차3법_2021년1월~2022년6월.xlsx'
data_path_2 = './data/임대차3법/임대차3법_2020년7월~12월.xlsx'
df_1 = pd.read_excel(data_path_1, sheet_name = 1, names = col_name, header = None)[2:]
df_2 = pd.read_excel(data_path_2, sheet_name = 1, names = col_name, header = None)[2:]
df = pd.concat([df_1, df_2], ignore_index = True)
# Cleaned dataframe
df = clean_dataset(df)
print(df.head())
# Save Cleaned DataFrame
df.to_csv('./data/clean_news_1.csv', mode = 'w')


# 1-2. Serious Accident Punishment Law
#col_name = ['Date', 'Paper', 'Review', 'Title', "Text"]
data_path_1 = './data/중대재해처벌법/중대재해처벌법_2022년3월~6월.xlsx'
data_path_2 = './data/중대재해처벌법/중대재해처벌법_2022년1월~2월.xlsx'
data_path_3 = './data/중대재해처벌법/중대재해처벌법_2021년1월~12월.xlsx'

df_1 = pd.read_excel(data_path_1, sheet_name = 1, names = col_name, header = None)[2:]
df_2 = pd.read_excel(data_path_2, sheet_name = 1, names = col_name, header = None)[2:]
df_3 = pd.read_excel(data_path_3, sheet_name = 1, names = col_name, header = None)[2:]
df = pd.concat([df_1, df_2, df_3], ignore_index = True)
# Cleaned dataframe
df = clean_dataset(df)
print(df.head())
# Save Cleaned DataFrame
df.to_csv('./data/clean_news_1.csv', mode = 'w')


# 1-3. Anti-discrimination Law
#col_name = ['Date', 'Paper', 'Review', 'Title', "Text"]
data_path_1 = './data/차별금지법/차별금지법_2022년4월~6월.xlsx'
data_path_2 = './data/차별금지법/차별금지법_2022년1월~3월.xlsx'
data_path_3 = './data/차별금지법/차별금지법_2021년10월~12월.xlsx'
data_path_4 = './data/차별금지법/차별금지법_2021년7월~9월.xlsx'
data_path_5 = './data/차별금지법/차별금지법_2021년5월~6월.xlsx'
data_path_6 = './data/차별금지법/차별금지법_2021년1월~4월.xlsx'
df_1 = pd.read_excel(data_path_1, sheet_name = 1, names = col_name, header = None)[2:]
df_2 = pd.read_excel(data_path_2, sheet_name = 1, names = col_name, header = None)[2:]
df_3 = pd.read_excel(data_path_3, sheet_name = 1, names = col_name, header = None)[2:]
df_4 = pd.read_excel(data_path_4, sheet_name = 1, names = col_name, header = None)[2:]
df_5 = pd.read_excel(data_path_5, sheet_name = 1, names = col_name, header = None)[2:]
df_6 = pd.read_excel(data_path_6, sheet_name = 1, names = col_name, header = None)[2:]
df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], ignore_index = True)
# Cleaned dataframe
df = clean_dataset(df)
print(df.head())
# Save Cleaned DataFrame
df.to_csv('./data/clean_news_1.csv', mode = 'w')


# 1-4. Carbon Neutral Law
#col_name = ['Date', 'Paper', 'Review', 'Title', "Text"]
data_path_1 = './data/탄소중립/탄소중립_2022년1월.xlsx'
data_path_2 = './data/탄소중립/탄소중립_2022년2월.xlsx'
data_path_3 = './data/탄소중립/탄소중립_2022년3월.xlsx'
data_path_4 = './data/탄소중립/탄소중립_2022년4월.xlsx'
data_path_5 = './data/탄소중립/탄소중립_2022년5월.xlsx'
data_path_6 = './data/탄소중립/탄소중립_2022년6월.xlsx'
df_1 = pd.read_excel(data_path_1, sheet_name = 1, names = col_name, header = None)[2:]
df_2 = pd.read_excel(data_path_2, sheet_name = 1, names = col_name, header = None)[2:]
df_3 = pd.read_excel(data_path_3, sheet_name = 1, names = col_name, header = None)[2:]
df_4 = pd.read_excel(data_path_4, sheet_name = 1, names = col_name, header = None)[2:]
df_5 = pd.read_excel(data_path_5, sheet_name = 1, names = col_name, header = None)[2:]
df_6 = pd.read_excel(data_path_6, sheet_name = 1, names = col_name, header = None)[2:]
df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], ignore_index = True)
# Cleaned dataframe
df = clean_dataset(df)
print(df.head())
# Save Cleaned DataFrame
df.to_csv('./data/clean_news_1.csv', mode = 'w')



# 2. Make Clean Korean News Corpus for pre-training
# 2-1. Load all cleaned dataframe
df_1 = pd.read_csv('./data/clean_news_1.csv')
df_2 = pd.read_csv('./data/clean_news_2.csv')
df_3 = pd.read_csv('./data/clean_news_3.csv')
df_4 = pd.read_csv('./data/clean_news_4.csv')

# 2-2. Extract Text from Text & Title columns
df_1_text = df_1.Text.to_list()
df_2_text = df_2.Text.to_list()
df_3_text = df_3.Text.to_list()
df_4_text = df_4.Text.to_list()

total_news = df_1_text + df_1_text + df_1_text + df_1_text
print("\n >> Korean News size : ", len(total_news))

# 2-3. Make clean News Corpus for pretraining
news_corpus = []
for news in total_news:
    news_sentences = news.split(".")
    for sent in news_sentences:
        news_corpus.append(sentence_split(sent))

news_corpus = [x for x in news_corpus if x != 'None']

print("\n >> Korean News Corpus size : ", len(news_corpus))

# 2-4. List to a File line by line
with open('./data/korean_news_corpus.txt', 'w') as fp:
    for row in news_corpus:
        fp.write("%s\n" % row)
    print("Write is Done.")
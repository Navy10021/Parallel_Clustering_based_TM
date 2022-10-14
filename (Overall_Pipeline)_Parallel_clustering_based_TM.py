from models.parallel_clustering_based_TM import *
from models.keyword_visualization import *


# 1. News article dataset(ex.중대재해 처벌법 관련 뉴스)
df = pd.read_csv('./data/clean_news_2.csv')
# 2. Load My pre-trained & fine-tuned Language model
my_model = './outputs/nil_sts_news-bert'
# Target columns text
target_column = 'Title'

# 3. Convert Text data to Sentence-level Embeddings
cluster = ParallelCluster(
    dataframe = df,
    tgt_col = target_column,
    model_name = my_model,
    use_sentence_bert = True
    )

# 4. Parallel Embeddings Clustering
clusters, unclusters = cluster.parallel_cluster(
    clusters = None,
    threshold = 0.75,    # Best threshold : Text(0.90), Title(0.75)
    page_size = 2500,
    iterations = 15
    )

# 5. Stack the cluseted text embeddings
col_list = ['Date','Paper', 'Review', 'Title', 'Text']

news_df = cluster.cluster_stack(
    col_list = col_list,
    clusters = clusters,
    unclusters = unclusters
    )

# 6. Topic Modeling : Extract Latent Topics (Top 50 Keywords)
top_n_words = cluster.extract_top_n_words_per_topic(
    dataframe = news_df,
    n = 50,
    en = False
    )

# 7. Parallel Clustering Results Table
news_df['Topic_Modeling'] = [top_n_words[i] for i in news_df['Topic'].values]
news_df.head(10)


# 8. Topic Modeling
news_keywords("중대재해처벌법", top_n_words, 10)  # (Topic name, Top K) 
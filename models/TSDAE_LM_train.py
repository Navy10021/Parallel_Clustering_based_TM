from models.TSDAE_LM import *


# 1. Load News Corpus data
file_path = "./data/korean_news_corpus.txt"
with open(file_path) as f:
    lines = f.readlines()
    sent_list = [line.rstrip('\n') for line in lines]

sent_list = [x for x in sent_list if x != 'None']
print(">> Total Number of Corpus for TSDAE training : {}".format(len(sent_list)))


# 2. Examples of Denoising AutoEncoder Dataset
train_data = DenoisingAutoEncoderDataset(sent_list[:10])
for row in train_data:
    print(row)
    
    
############################################
# 3.TSDAE unsupervised-embeddings training #
############################################

# 3-1. Dataset with noise function
train_data = DenoisingAutoEncoderDataset(sent_list)

# 3-2. Dataloader
loader = DataLoader(
    train_data,
    batch_size = 8,
    shuffle = True, 
    drop_last = True
    )

# 3-3. Retrain or not / Save my model or not
re_train = True
save_MyModel = True

# 3-4-1. Load trained model 
if re_train:
    print("\n *** Retrain model *** \n")
    model_name = './outputs/news-bert/20000'
    model = SentenceTransformer(model_name)
# 3-4-2. Sentence Embedding using [CLS] token or Mean/Max Pooling
else:
    print("\n *** Pretrain model *** \n")
    model_name = 'klue/bert-base' 
    bert = models.Transformer(model_name)
    pooling = models.Pooling(bert.get_word_embedding_dimension(), 'mean') # cls, mean, max
    model = SentenceTransformer(modules = [bert, pooling])

# 3-5. Use Loss function
loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder = True)

epochs = 1      # Best epoch == 3
warmup_steps = int(len(loader) * epochs * 0.05) # Warmup 5 %

# 3-6. Train
model.fit(
    train_objectives=[(loader, loss)],
    epochs = epochs,
    warmup_steps = warmup_steps,
    checkpoint_path = './outputs/news-bert',
    checkpoint_save_steps = 20000,
    weight_decay = 0,  
    scheduler = 'constantlr',
    optimizer_params = {'lr': 3e-5},
    show_progress_bar = True
)

# 3-7. Save final model
if save_MyModel:
    model.save('./outputs/news-bert')
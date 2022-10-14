from models.Sentence_embedding import *

#####################
## Korean NLI Task ##
#####################
# 1. Load Language models
model_path = './outputs/news-bert'
model = SentenceTransformer(model_path)

# 2. Load dataset
train_snli = pd.read_csv("data/snli_1.0_train.ko.tsv", sep='\t', quoting=3)  
train_xnli = pd.read_csv("data/multinli.train.ko.tsv", sep='\t', quoting=3)
train_data = pd.concat([train_snli, train_xnli], ignore_index=True)
print(">> Total Train Dataset size :", len(train_data))

# Dataset for Eval
val_data = pd.read_csv("data/sts-dev.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("data/sts-test.tsv", sep='\t', quoting=3)
print(">> Total Validataion Dataset size :", len(val_data))
print(">> Total Test Dataset size :", len(test_data))

# label_dict
label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}

train_data = drop_kornli(train_data)
print(train_data.head())

# 3. Make Dataset for NLI Task Training
# 3-1. Traing dataset
train_batch_size = 16
train_samples = make_kornli_dataset(train_data)

# 3-2. Train DataLoader
train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

# 3-3. Val/Test dataset
val_data = drop_korsts(val_data)
test_data = drop_korsts(test_data)

dev_samples = make_korsts_dataset(val_data)
test_samples = make_korsts_dataset(test_data)

# 3-4. Eval DataLoader
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# 3-5. Loss function : Calculate MSE loss
train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=len(label_dict))

# 3-6. Warmup(10% of train data for warm-up) & Epochs
num_epochs = 3
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) # 10%
model_save_path = './outputs/nil_news-bert'

# 4. Training
model.fit(train_objectives = [(train_dataloader, train_loss)],
          evaluator = dev_evaluator,
          epochs = num_epochs,
          evaluation_steps = 1000,
          warmup_steps = warmup_steps,
          output_path = model_save_path)


#####################
## Korean STS Task ##
#####################

# 1. Load dataset
train_batch_size = 16
train_data = pd.read_csv("data/sts-train.tsv", sep='\t', quoting=3)
val_data = pd.read_csv("data/sts-dev.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("data/sts-test.tsv", sep='\t', quoting=3)

train_data = drop_korsts(train_data)
val_data = drop_korsts(val_data)
test_data = drop_korsts(test_data)

# 2. Make Dataset for STS Task Training
train_samples = make_korsts_dataset(train_data)
dev_samples = make_korsts_dataset(val_data)
test_samples = make_korsts_dataset(test_data)

# 2-1. DataLoader
train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

# 2-3. Loss function : Calculate Cosine similarity
train_loss = losses.CosineSimilarityLoss(model=model)

# 2-4. Evaluator 
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# 2-5. Warmup(10% of train data for warm-up) & Epochs
model_save_path = './outputs/nil_sts_news-bert'
num_epochs = 15
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)  

# 3. Training
model.fit(train_objectives = [(train_dataloader, train_loss)],
          evaluator = evaluator,
          epochs = num_epochs,
          evaluation_steps = 1000,
          warmup_steps = warmup_steps,
          output_path = model_save_path)


# 4. Evaluation
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
print(">> Best TEST Socre is : {:.4f}".format(test_evaluator(model, output_path=model_save_path)))
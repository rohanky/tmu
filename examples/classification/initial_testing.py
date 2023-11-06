from datasets import load_dataset
import pandas as pd
import numpy as np
import string
import nltk
import logging
import argparse
import random
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.classification.one_vs_one_classifier import TMOneVsOneClassifier
from tmu.tools import BenchmarkTimer

# Set the random seed
random_seed = 42  # You can use any integer as the seed
random.seed(random_seed)

_LOGGER = logging.getLogger(__name__)


nltk.download('punkt')  # Download the NLTK tokenization model if you haven't already

alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


train_dataset = load_dataset('trec', split='train')
test_dataset = load_dataset('trec', split='test')

df_train = pd.DataFrame(train_dataset)
df_test = pd.DataFrame(test_dataset)


from sklearn.model_selection import train_test_split

def create_class_subset(df, class_column, split_ratio=0.05, random_state=42):
    unique_labels = df[class_column].unique()
    subset_df = pd.DataFrame(columns=df.columns)

    for label in unique_labels:
        class_df = df[df[class_column] == label]
        _, class_subset = train_test_split(class_df, test_size=split_ratio, random_state=random_state)
        subset_df = pd.concat([subset_df, class_subset])

    return subset_df

# Example usage:
#df_train_sub = create_class_subset(df_train, class_column='label')
#df_test_sub = create_class_subset(df_test, class_column='label')


df_train_sub = df_train
df_test_sub = df_test

print(df_train_sub.shape)
print(df_test_sub.shape)



def preprocess_text(text):
    tokens = word_tokenize(text)
    #tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stopwords and word.lower() not in alpha]
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower()]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return tokens


df_train_sub['text_processed'] = df_train_sub['text'].apply(preprocess_text)
df_test_sub['text_processed'] = df_test_sub['text'].apply(preprocess_text)
#df_train_sub = df_train_sub.sample(frac=1, random_state=42)  # Use a random_state for reproducibility
#df_test_sub = df_test_sub.sample(frac=1, random_state=42)  # Use a random_state for reproducibility

'''# Create a mapping of unique topic names to unique numeric values
unique_topics = df_train_sub['label'].unique()
topic_mapping = {topic: i for i, topic in enumerate(unique_topics)}
df_train_sub['label_num'] = df_train_sub['label'].map(topic_mapping)
df_test_sub['label_num'] = df_test_sub['label'].map(topic_mapping)'''
Y_train = df_train_sub['coarse_label'].values
Y_test = df_test_sub['coarse_label'].values


# Define the available labels
available_labels = list(set(Y_train))
# Define the percentage of noise you want (e.g., 10%)
percentage_noise = 50

# Calculate the number of elements to change
num_elements_to_change = int(len(Y_train) * percentage_noise / 100)

# Generate random indices to introduce noise
indices_to_change = random.sample(range(len(Y_train)), num_elements_to_change)

# Introduce noise by changing labels to other available labels
Y_train_noisy = Y_train.copy()  # Create a copy to avoid modifying the original array
for index in indices_to_change:
    current_label = Y_train_noisy[index]
    available_labels.remove(current_label)  # Remove the current label from available labels
    new_label = random.choice(available_labels)  # Choose a new label from the available labels
    Y_train_noisy[index] = new_label
    available_labels.append(current_label)  # Put the old label back into available labels

different_elements = np.sum(Y_train != Y_train_noisy)
print(different_elements)
print(Y_train[0:20])
print(Y_train_noisy[0:20])







def create_vocab(df):
    input_text = df['text_processed'].tolist()
    vocab = []
    for i in input_text:
        
        for j in i:
            vocab.append(j)
    print('full vocab', len(vocab))
    fdist1 = FreqDist(vocab)
    tokens1 = fdist1.most_common(4000)
    full_token_fil = []
    for i in tokens1:
        full_token_fil.append(i[0])

    vocab_unique = full_token_fil

    sum1 = 0
    for j in tokens1:
        sum1 += j[1]

    print('sum1', sum1)
    
    return vocab_unique


vocab_selected = create_vocab(df_train_sub)



def binarization_text(df):
    feature_set = np.zeros([len(df), len(vocab_selected)], dtype=np.uint8)
    tnum=0
    for t in df:
        for w in t:
            if (w in vocab_selected):

                idx = vocab_selected.index(w)
                feature_set[tnum][idx] = 1
        tnum += 1
    return feature_set

input_text_train = df_train_sub['text_processed'].tolist()
input_text_test = df_test_sub['text_processed'].tolist()



X_train = binarization_text(input_text_train)
X_test = binarization_text(input_text_test)


# Create masks for clean and noisy samples based on the equality condition
clean_mask = (Y_train == Y_train_noisy)
noisy_mask = (Y_train != Y_train_noisy)

# Separate X_train into clean and noisy based on the masks
X_train_clean = X_train[clean_mask]
X_train_noisy = X_train[noisy_mask]
    
print(X_train.shape)
print(X_test.shape)

def normalize_rows(arr):
    # Calculate the result for each row (normalize based on local maximum and minimum, then scale to sum to 1)
    max_values = np.max(arr, axis=1, keepdims=True)
    min_values = np.min(arr, axis=1, keepdims=True)

    normalized_scores = (arr - min_values) / (max_values - min_values)
    normalized_scores /= normalized_scores.sum(axis=1, keepdims=True)

    return normalized_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=1500, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=False, type=bool)
    parser.add_argument("--epochs", default=10, type=int)

    args = parser.parse_args()

    tm = TMOneVsOneClassifier(args.num_clauses, args.T, args.s, platform=args.device, weighted_clauses=args.weighted_clauses)

    _LOGGER.info(f"Running {TMOneVsOneClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, Y_train_noisy)

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(X_test) == Y_test).mean()

            y_pred = tm.predict(X_test)
            from sklearn.metrics import f1_score

            f1 = f1_score(Y_test, y_pred, average = 'macro')
            

            

        _LOGGER.info(f"Epoch: {epoch + 1}, Testing_Accuracy: {result:.2f}, F1_Score: {f1*100:.2f},  Training Time: {benchmark1.elapsed():.2f}s, "
                    f"Testing Time: {benchmark2.elapsed():.2f}s")
       

    clause_scores_clean = tm.get_confidence_score(X_train_clean)
    clause_scores_noisy = tm.get_confidence_score(X_train_noisy)

 

    #result_clean = np.max(clause_scores_clean, axis=1) - (np.sum(clause_scores_clean, axis=1) - np.max(clause_scores_clean, axis=1))
    #result_noisy = np.max(clause_scores_noisy, axis=1) - (np.sum(clause_scores_noisy, axis=1) - np.max(clause_scores_noisy, axis=1))

    #result_clean_norm = normalize_rows(clause_scores_clean)
    #result_noisy_norm = normalize_rows(clause_scores_noisy)

    result_clean = 12000-np.max(clause_scores_clean, axis=1)
    result_noisy = 12000-np.max(clause_scores_noisy, axis=1)

    df_clean = pd.DataFrame(result_clean)
    df_clean.to_csv("data_clean.csv")


    df_noisy= pd.DataFrame(result_noisy)
    df_noisy.to_csv("data_noisy.csv")

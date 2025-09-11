import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def predict_email_class(email, spam_word_prob, non_spam_word_prob, P_spam, P_non_spam, word_to_index):
    log_prob_spam = np.log(P_spam)
    log_prob_non_spam = np.log(P_non_spam)
    words = email.split()
    indices = []
    for word in words:
        if word in word_to_index:
            indices.append(word_to_index[word])
    spam_log_probs = np.log(spam_word_prob[indices])
    non_spam_log_probs = np.log(non_spam_word_prob[indices])
    log_prob_spam += np.sum(spam_log_probs)
    log_prob_non_spam += np.sum(non_spam_log_probs)
    if log_prob_spam > log_prob_non_spam:
        return 1
    else:
        return 0
    
def accuracy_score(y_true, y_pred):
    total_samples = len(y_true)
    correct_predictions = 0
    for i in range(total_samples):
        if y_true.iloc[i] == y_pred[i]:
            correct_predictions += 1
    accuracy = correct_predictions / total_samples
    return accuracy

file = pd.read_csv('emails.csv')
if 'text' in file.columns:
    file['text'] = file['text'].apply(preprocess_text)
train1, test1, train2, test2 = train_test_split(file['text'], file['spam'], test_size=0.5, random_state=532)
vectorizer = CountVectorizer()
train1_bow = vectorizer.fit_transform(train1)
test1_bow = vectorizer.transform(test1)
features = vectorizer.get_feature_names_out()
number_of_features = len(features)
spam_counts = np.zeros(number_of_features)
non_spam_counts = np.zeros(number_of_features)
for i in range(len(train2)):
    if train2.iloc[i] == 1:
        spam_counts += train1_bow[i].toarray()[0]
    else:
        non_spam_counts += train1_bow[i].toarray()[0]
spam_word_prob = (spam_counts + 1) / (np.sum(spam_counts) + number_of_features)
non_spam_word_prob = (non_spam_counts + 1) / (np.sum(non_spam_counts) + number_of_features)
P_spam = sum(train2) / len(train2)
P_non_spam = 1 - P_spam
word_to_index = {word: i for i, word in enumerate(features)}
predictions = [predict_email_class(email, spam_word_prob, non_spam_word_prob, P_spam, P_non_spam, word_to_index) for email in test1]
accuracy = accuracy_score(test2, predictions)
print(f"Accuracy of the model: {accuracy:.8f}")

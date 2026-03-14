import pandas as pd
import nltk
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from preprocessing import preprocess_text

df1 = pd.read_csv("data/spam.csv",encoding="latin-1")
df2 = pd.read_csv("data/usermessages.csv")
df = pd.concat([df1, df2], ignore_index=True)

df = df[['v1','v2']]
df.columns = ['label','message']
df['label'] = df['label'].map({'ham':0,'spam':1})
df['message'] = df['message'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['message'],df['label'],test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.95)
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vector,y_train)

pred = model.predict(X_test_vector)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("Model saved successfully!")

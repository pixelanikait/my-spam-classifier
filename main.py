
import pickle
import os
import csv
import pandas as pd

from preprocessing import preprocess_text

def explain_prediction(msg, vectorizer, model):
    words = msg.split()

    feature_names = vectorizer.get_feature_names_out()
    vocab = vectorizer.vocabulary_

    spam_weights = model.feature_log_prob_[1]
    ham_weights = model.feature_log_prob_[0]

    spam_words = []
    ham_words = []

    for word in words:
        if word in vocab:
            idx = vocab[word]
            score = spam_weights[idx] - ham_weights[idx]

            if score > 0:
                spam_words.append((word, score))
            else:
                ham_words.append((word, score))

    spam_words.sort(key=lambda x: x[1], reverse=True)
    ham_words.sort(key=lambda x: x[1])

    print("\nTop spam indicators:")
    for word, score in spam_words[:5]:
        print(f"{word} (+{score:.3f})")

    print("\nHam indicators:")
    for word, score in ham_words[:5]:
        print(f"{word} ({score:.3f})")

model = pickle.load(open("model/model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

msg1 = input("Enter message: ")
msg = preprocess_text(msg1)
print("Processed:", msg)

msg_vec = vectorizer.transform([msg])
pred = model.predict(msg_vec)

prob = model.predict_proba(msg_vec)[0][1]
print("Spam probability:", round(prob,3))

if pred[0] == 1:
    print("Spam")
else:
    print("Not spam")

explain_prediction(msg, vectorizer, model)

label = input("Is this actually spam? (y/n): ").lower()
true_label = "ham" if label == "n" else "spam"

with open("data/usermessages.csv","a",newline="",encoding="latin-1") as f:
    writer = csv.writer(f)
    writer.writerow([true_label,msg1,"","",""])

if (pred[0] == 1 and true_label == "ham") or (pred[0] == 0 and true_label == "spam"):
    print("Updating model with new example...")
    os.system("python train.py")


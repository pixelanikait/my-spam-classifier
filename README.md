## Overview:
Spam classifier:
An ML mini project that classifies messages as spam or not spam

Uses python, scikit-learn, pandas, NLTK

Model is a Multinomial naive-bayes with TF-IDF vectorization

## Installation:

`git clone https://github.com/pixelanikait/my-spam-classifier.git`

`cd spam-classifier`

`pip install -r requirements.txt`

## Usage:
1. Train model: `python train.py`
2. Run classifier: `python main.py`

## Example Output:
```text
Enter message: Congratulations!!! You won a chance to get a free car 
Processed: congratul won chanc get free car 
Spam probability: 0.591 
Spam 

Top spam indicators: 
won (+2.942) 
congratul (+1.652) 
free (+1.227) 
chanc (+1.149) 

Ham indicators: 
car (-1.625) 
get (-1.249) 
Is this actually spam? (y/n): y
```

data set from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

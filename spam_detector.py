# spam_detector.py

import pandas as pd
import string
import nltk
import warnings
warnings.filterwarnings("ignore")  # optional: suppress sklearn warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional: download stopwords silently
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# 1. Load dataset
df = pd.read_csv("SMSSpamCollection.txt", sep='\t', names=["label", "message"])
print("✅ Dataset Loaded. First 5 rows:\n")
print(df.head())

# 2. Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

print("\n🔄 Preprocessing text...")
df['cleaned'] = df['message'].apply(preprocess_text)

# 3. Feature extraction
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned'])
y = df['label']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Predict & evaluate
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

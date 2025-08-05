# Spam Mail Detector

This project is a simple spam message classifier using the SMS Spam Collection dataset from the UCI repository.

## 📂 Dataset

The dataset contains 5574 labeled SMS messages as spam or ham.

- Downloaded from: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## 🚀 Features

- Text preprocessing (cleaning, stopword removal)
- TF-IDF vectorization
- Classification using Multinomial Naive Bayes
- Evaluation with accuracy, F1 score, confusion matrix

## 🛠 Technologies

- Python
- pandas
- scikit-learn
- nltk

## 🧪 How to Run

```bash
pip install -r requirements.txt
python spam_detector.py

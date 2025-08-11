# 📧 Spam Mail Detector

A machine learning project that classifies SMS messages as **spam** or **ham** (not spam) using the **SMS Spam Collection dataset** from the UCI repository.  
This project demonstrates **text preprocessing**, **feature extraction**, and **classification** using **Multinomial Naive Bayes** and other models.

---

## 📂 Dataset

- **Total messages:** 5,574 labeled SMS messages  
- **Labels:** `spam` or `ham`  
- **Source:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)  

---

## 🚀 Features

- **Text preprocessing**: Cleaning, lowercasing, tokenization, and stopword removal  
- **TF-IDF vectorization** for converting text into numerical features  
- **Classification** using:
  - Multinomial Naive Bayes (primary)
  - Logistic Regression / SVM (optional for comparison)
- **Evaluation metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix visualization

---

## 🛠 Technologies

- **Language:** Python  
- **Libraries:**
  - `pandas`
  - `scikit-learn`
  - `nltk`
  - `matplotlib` / `seaborn` (for visualization)

---

## 🧪 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vaishnavilekawale/SpamDetectorProject.git
cd SpamDetectorProject

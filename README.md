# 📰 Fake News Detection using NLP

## 📌 Project Overview
This project aims to classify news articles as **real or fake** using machine learning algorithms. It utilizes **TF-IDF vectorization** for feature extraction and evaluates multiple classification models to determine the most accurate one.

## 🚀 Features
- Data preprocessing including stopword removal and tokenization
- Implementation of various machine learning models:
  - **Multinomial Naive Bayes**
  - **Decision Tree**
  - **Random Forest**
  - **Passive Aggressive Classifier**
  - **Logistic Regression**
- Evaluation metrics: **Accuracy, Precision, Recall, F1-score**
- Visualization of model performance using Matplotlib & Seaborn
- A function to predict whether a given news title is **fake** or **real**

## 📂 Dataset
The dataset used for training and testing consists of real and fake news articles. The text data is transformed using **TF-IDF vectorization** to convert it into numerical format.

## 📑 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/itsaman080/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download NLTK Resources
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 4️⃣ Run the Project
```bash
python main.py
```

## 📊 Model Evaluation
Each model is evaluated using **Accuracy, Precision, Recall, and F1-score**:
| Model                      | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| Naive Bayes                | XX%      | XX%       | XX%    | XX%      |
| Decision Tree              | XX%      | XX%       | XX%    | XX%      |
| Random Forest              | XX%      | XX%       | XX%    | XX%      |
| Passive Aggressive Classifier | XX%  | XX%       | XX%    | XX%      |
| Logistic Regression        | XX%      | XX%       | XX%    | XX%      |

## 📌 Example Usage
```python
from prediction import predict_title

# Example Titles
title_text_1 = "Donald Trump Sends Out Embarrassing New Year"
title_text_2 = "As U.S. budget fight looms, Republicans flip their fiscal script"

# Predictions
print(predict_title(title_text_1))  # Output: Fake or Real
print(predict_title(title_text_2))  # Output: Fake or Real
```

- [LinkedIn](https://www.linkedin.com/in/aman-kumar-thakur/)
- [Github](https://github.com/itsaman080)

---


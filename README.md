# 📧 Spam Email Classification

An end-to-end spam email classification project using **Natural Language Processing (NLP)** and **classical machine learning algorithms**.  
The project focuses on achieving **high precision** to minimize false positives while maintaining strong overall performance.

---

## 📂 Repository Structure

```

Spam_Email_Classification/
│
├── README.md
├── SpamEmailClassificationNotebook.ipynb
├── SpamEmailClassificationReport.pdf
├── emails.csv

```

---

## 🎯 Objective

To build and evaluate machine learning models for classifying email messages as **Spam** or **Ham (Not Spam)** based on their textual content.  
Precision is prioritized because misclassifying legitimate emails as spam is more costly than allowing some spam emails to pass through.

---

## 📊 Dataset Information

- **Dataset Name:** Spam Email Dataset  
- **Source:** Kaggle 
- **Target Variable:**
  - `1` → Spam  
  - `0` → Ham  

### Initial Columns
- `text`: Raw email content  
- `spam`: Spam label  

### Data Cleaning Summary
- Retained only relevant columns
- Removed missing and duplicate values
- Removed invalid labels
- Final dataset size: **5,693 emails**
- Converted labels to numeric format

The dataset is **imbalanced**, with ham emails significantly outnumbering spam emails.

---

## 🔍 Exploratory Data Analysis (EDA)

- Engineered length-based features:
  - Number of characters
  - Number of words
  - Number of sentences
- Spam and ham emails show **positively skewed distributions**
- Significant overlap exists between classes
- Strong correlation among length features
- No strong linear relationship between email length and spam label

Conclusion: **Length-based features alone are insufficient for classification**

---

## 🧹 Text Preprocessing

The following preprocessing steps were applied:
- Removal of subject prefixes
- Lowercasing
- Tokenization (NLTK)
- Removal of non-alphanumeric tokens
- Stopword and punctuation removal
- Stemming using Porter Stemmer

A new column `transformed_text` was created for modeling.

---

## 🧠 Feature Extraction

- **TF-IDF Vectorization**
  - Maximum features: 3000
  - Vectorizer fitted only on training data to prevent data leakage
- **Min–Max Scaling** applied to TF-IDF features

---

## 🤖 Models Trained

- Gaussian Naive Bayes  
- Multinomial Naive Bayes  
- Bernoulli Naive Bayes  
- Complement Naive Bayes  
- Logistic Regression  
- Support Vector Classifier  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Bagging  
- Extra Trees  
- Gradient Boosting  
- XGBoost  

Hyperparameter tuning was performed using **GridSearchCV** with **Stratified K-Fold Cross-Validation (k = 4)**.  
**Precision** was used as the primary scoring metric.

---

## 🔗 Ensemble Models

### Voting Classifier
- Base models: Multinomial NB, Gradient Boosting, XGBoost  
- Soft voting strategy  

### Stacking Classifier
- Base learners: Multinomial NB, Gradient Boosting, XGBoost  
- Meta-learner: Logistic Regression  

---

## 🏆 Final Model Performance

**Stacking Classifier**
- Accuracy: **0.9895**
- Precision: **0.9712**
- F1-score: **0.9783**
- ROC–AUC: **≈ 1.00**

The ROC curve lies close to the top-left corner, indicating strong discriminative capability.

---

## 🛠️ Tools & Libraries Used

- **Python**
- NumPy
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- XGBoost
- WordCloud

---

## 🔮 Conclusion

The Stacking Classifier achieved the best overall performance by effectively balancing precision and recall. Due to class imbalance, accuracy alone was found to be misleading; therefore, precision and F1-score were emphasized for model comparison.
While TF-IDF-based models perform well, they ignore word order and semantic context, rely on a fixed vocabulary, and do not handle concept drift or multilingual data. Important metadata such as email headers and URLs were not incorporated.

Future work may explore alternative embeddings, deep learning models, improved imbalance handling, and adaptive learning strategies.

---

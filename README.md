# 📧 Spam Email Classification

## 📊 About the Dataset

### 🗂 Dataset Name

**Spam Email Dataset**

### 📝 Description

This dataset consists of email text messages labeled as either **spam** or **not spam**. Each email is associated with a binary target variable, making the dataset suitable for supervised learning tasks in spam detection and text classification.

### 📑 Columns

* **text** — Contains the textual content of the email messages, including the body and headers
* **spam_or_not** — Binary target variable

  * `1` → Spam
  * `0` → Not Spam

### 🎯 Usage

The dataset is used for **Natural Language Processing (NLP)** tasks, specifically for building and evaluating spam email classification models.

---

## 🔄 Workflow

The notebook follows a structured machine learning pipeline:

1. 🧹 Data Cleaning
2. 📊 Exploratory Data Analysis (EDA)
3. 🧽 Text Preprocessing
4. 🔡 Feature Engineering
5. 🤖 Model Building and Evaluation
6. 📈 Model Comparison and Selection

---

## 🧹 Data Cleaning

* Dataset structure is inspected using `info()`
* Column names are verified
* Dataset shape is examined

These steps ensure the dataset is properly understood before further analysis.

---

## 📈 Exploratory Data Analysis (EDA)

Basic exploratory analysis is performed to understand the structure of the dataset and prepare it for preprocessing and modeling.

---

## 🧽 Text Preprocessing

The following preprocessing steps are applied to the email text:

* Conversion to lowercase
* Tokenization
* Removal of special characters
* Removal of stopwords and punctuation
* Stemming

These steps help reduce noise and standardize the text data before feature extraction.

---

## 🔡 Feature Engineering

* Email text is transformed into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**
* TF-IDF assigns higher weights to informative words while reducing the impact of frequently occurring but less meaningful terms

This representation enables traditional machine learning models to process textual data effectively.

---

## 🤖 Model Building & Evaluation

* Multiple machine learning models are trained and evaluated on the processed data
* Model performance is assessed using:

  * Accuracy
  * Precision
  * F1-score
  * ROC Curve Analysis (Only for the final selected model)

Given the application context, **precision is prioritised**, as misclassifying legitimate emails as spam is more costly than allowing some spam emails into the inbox.

---

## 🏆 Final Conclusion

Since misclassifying legitimate emails as spam is more costly than allowing some spam emails into the inbox, **precision is prioritised** in this task. Among all the evaluated models, **Multinomial Naive Bayes** achieves the highest precision (**0.9918**) while also maintaining strong **test accuracy (0.9858)** and **F1 score (0.9757)**. This balance makes it the most suitable model for the spam email classification problem.

The current approach relies on **TF-IDF features** and **Multinomial Naive Bayes**. TF-IDF ignores word order and semantic context, which limits its ability to capture meaning conveyed through phrases, sarcasm, or sentence structure. The model depends on a fixed vocabulary learned during training and may struggle with **evolving spam strategies, newly emerging terms (concept drift), and generalisation to other languages**. Additionally, Multinomial Naive Bayes assumes **conditional independence between words**, an assumption often violated in natural language, which can restrict performance on more complex email patterns.

Future work can explore **broader hyperparameter tuning**, **deep learning approaches**, **alternative word embeddings**, **class imbalance handling techniques such as SMOTE and cost-sensitive learning**, and **adaptive learning strategies** to improve robustness and generalisation.

---

## 🛠️ Tools & Packages Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* XGBoost

---

## 👩‍💻 Author

**Sohini Mandal**

---

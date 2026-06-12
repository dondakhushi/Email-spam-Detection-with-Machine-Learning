# 📧 Email Spam Detection using Machine Learning

## 📌 Project Overview

Email Spam Detection is a Machine Learning project that classifies emails as **Spam** or **Not Spam (Ham)**. Spam emails often contain advertisements, scams, phishing links, or unwanted content. This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically identify spam messages.

The model is trained on a labeled email dataset and uses **TF-IDF Vectorization** for feature extraction and **Multinomial Naive Bayes** for classification.

---

## 🎯 Objectives

* Analyze email text data.
* Preprocess and clean textual information.
* Convert text into numerical features using TF-IDF.
* Train a Machine Learning model to classify emails.
* Evaluate model performance using various metrics.
* Predict whether a custom email is spam or not.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Natural Language Processing (NLP)

---

## 📂 Project Structure

```text
Email-spam-Detection-with-Machine-Learning/
│
├── spam.csv
├── Email-spam-Detection-with-Machine-Learning.py
├── requirements.txt
├── README.md
```

---

## 📊 Dataset

The dataset contains email messages labeled as:

* **Ham (0)** → Legitimate Email
* **Spam (1)** → Unwanted or Fraudulent Email

Dataset Features:

| Column  | Description               |
| ------- | ------------------------- |
| label   | Email Category (Spam/Ham) |
| message | Email Content             |

---

## ⚙️ Machine Learning Workflow

### 1. Data Collection

Load the spam email dataset.

### 2. Data Cleaning

* Remove unnecessary columns.
* Handle missing values.
* Remove duplicate records.

### 3. Data Visualization

* Analyze spam and ham distributions.
* Visualize class balance.

### 4. Text Processing

Convert email text into numerical features using TF-IDF Vectorization.

### 5. Model Training

Train a Multinomial Naive Bayes classifier.

### 6. Model Evaluation

Evaluate using:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 7. Prediction

Classify new custom email messages.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/dondakhushi/Email-spam-Detection-with-Machine-Learning.git
```

### Move into Project Directory

```bash
cd Email-spam-Detection-with-Machine-Learning
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python Email-spam-Detection-with-Machine-Learning.py
```

---

## 📈 Sample Results

```text
Accuracy: 98.3%

Spam Detection Successful
```

The model can accurately classify spam and legitimate emails using text-based features.

---

## 🔍 Example Prediction

Input:

```text
Congratulations! You have won a free iPhone. Click here to claim now.
```

Output:

```text
SPAM EMAIL
```

Input:

```text
Meeting scheduled tomorrow at 10 AM. Please join on time.
```

Output:

```text
NOT SPAM EMAIL
```

---

## 🌟 Future Improvements

* Deploy using Flask or Streamlit.
* Use Deep Learning (LSTM/RNN).
* Add advanced text preprocessing.
* Support multiple languages.
* Build a web-based spam detection application.

---

## 📚 Learning Outcomes

Through this project, you will learn:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Natural Language Processing
* Feature Engineering
* TF-IDF Vectorization
* Machine Learning Classification
* Model Evaluation

---

## 👨‍💻 Author

Khushi Donda

Machine Learning & Data Science Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it with others.

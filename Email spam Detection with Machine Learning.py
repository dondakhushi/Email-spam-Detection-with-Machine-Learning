
# EMAIL SPAM DETECTION USING MACHINE LEARNING

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. LOAD DATASET


# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())


# 2. DATA CLEANING


# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("\nColumns after renaming:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()


# 3. DATA VISUALIZATION

# Count spam and ham emails
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Emails")
plt.xlabel("Email Type")
plt.ylabel("Count")
plt.show()

# 4. LABEL ENCODING

# Convert labels to numeric
# ham = 0, spam = 1

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nEncoded Labels:")
print(df.head())

# 5. SPLIT DATA

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 6. TEXT VECTORIZATION (TF-IDF)


vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Transformation Completed")


# 7. TRAIN MODEL


model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("\nModel Training Completed")


# 8. PREDICTIONS


y_pred = model.predict(X_test_tfidf)


# 9. MODEL EVALUATION


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10. CONFUSION MATRIX


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 11. CUSTOM EMAIL TESTING

while True:

    user_email = input("\nEnter Email Text (or type 'exit'): ")

    if user_email.lower() == "exit":
        print("Program Ended.")
        break

    email_data = vectorizer.transform([user_email])

    prediction = model.predict(email_data)

    if prediction[0] == 1:
        print("Result: SPAM EMAIL")
    else:
        print("Result: NOT SPAM EMAIL")
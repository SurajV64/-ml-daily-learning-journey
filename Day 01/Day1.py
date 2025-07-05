# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Sample dataset (ham = not spam, spam = spam)
data = {
    'text': [
        'Congratulations! You won a free iPhone. Click here to claim.',
        'Hi, how are you doing today?',
        'Free entry in a contest! Win cash now!',
        'Are you coming to the meeting tomorrow?',
        'You have been selected for a free lottery prize!',
        'Let’s catch up soon, it’s been a while!',
        'Get cheap medicine online at low price.',
        'Lunch at 1pm?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Step 3: Convert text to numeric features (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Step 4: Encode labels (spam=1, ham=0)
y = df['label'].map({'ham': 0, 'spam': 1})

# Step 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Step 6: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Step 8: Test with custom input
def predict_spam(message):
    input_data = vectorizer.transform([message])
    prediction = model.predict(input_data)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Example test
print(predict_spam("Free vacation to Bahamas!!!"))
print(predict_spam("Hey Chinnu, can you send the notes?"))

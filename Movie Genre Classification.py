import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load and Parse the Text File
file_path =  r"C:\Users\SHIVAKUMAR\Desktop\PythonPrograms\internship\movie genre test data.txt"

ids = []
titles = []
descriptions = []
genres = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(" ::: ")
        if len(parts) == 3:
            movie_id = parts[0].strip()
            title = parts[1].split('(')[0].strip()
            description = parts[2].strip()

            # Simulate genre assignment (demo purpose only)
            if 'love' in description.lower():
                genre = ['Romance']
            elif 'murder' in description.lower() or 'crime' in description.lower():
                genre = ['Crime']
            else:
                genre = ['Drama']

            ids.append(movie_id)
            titles.append(title)
            descriptions.append(description)
            genres.append(genre)

# Step 2: Create DataFrame
df = pd.DataFrame({
    'ID': ids,
    'TITLE': titles,
    'DESCRIPTION': descriptions,
    'GENRE': genres
})

# Step 3: Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['DESCRIPTION'])

# Step 4: Encode Genre Labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['GENRE'])

# Step 5: Train-Test Split with index tracking
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, df.index, test_size=0.3, random_state=42
)

# Step 6: Train the Model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Step 7: Predictions and Evaluation
y_pred = model.predict(X_test)
predicted_genres = mlb.inverse_transform(y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=mlb.classes_))

# Step 8: Show predictions with ID, Title, Description, and Predicted Genre
predicted_df = pd.DataFrame({
    'ID': df.loc[test_idx, 'ID'].values,
    'TITLE': df.loc[test_idx, 'TITLE'].values,
    'DESCRIPTION': df.loc[test_idx, 'DESCRIPTION'].values,
    'GENRE': [', '.join(genres) for genres in predicted_genres]
})

print("\nSample Predictions:")
print(predicted_df.head(100))


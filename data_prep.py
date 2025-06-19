import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions

csv_file = r"C:\Users\imman\.vscode\Code\Python\USA\Final project\dataset.csv"

df = pd.read_csv(csv_file)

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

stop_words = set(stopwords.words('english'))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words and len(word) > 2]))

lemmatizer = WordNetLemmatizer()
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

df.to_csv("preprocessed.csv", index=False)
print("Data preprocessing complete!")
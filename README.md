NLP Text Normalization: Lemmatization and Stemming

This repository demonstrates how to perform lemmatization and stemming in Natural Language Processing (NLP) using Python libraries like NLTK and spaCy.

📌 Overview

. Text normalization is a crucial preprocessing step in NLP. It helps in reducing words to their root or base form, which improves the efficiency and accuracy of downstream tasks such as text classification, sentiment analysis, and search.

. Stemming: Reduces words to their stem (e.g., playing → play). It is a rule-based, faster method but can sometimes produce non-dictionary words.

. Lemmatization: Reduces words to their dictionary form (lemma) using linguistic knowledge (e.g., better → good). It is more accurate but computationally heavier.

🛠️ Requirements



Install the following dependencies before running the code:

pip install nltk spacy


Download required resources:

import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


For spaCy lemmatization:

python -m spacy download en_core_web_sm

📂 Project Structure
├── stemming_vs_lemmatization.ipynb   # Notebook with examples
├── requirements.txt                  # List of dependencies
└── README.md                         # Project documentation

🚀 Usage

Run the Jupyter notebook or Python script to see examples of:

Tokenization of sentences

Applying different stemmers (Porter, Snowball)

Applying lemmatization with NLTK WordNetLemmatizer and spaCy

Example snippet:

from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["playing", "better", "studies", "happiest"]

print("Stemming:")
print([stemmer.stem(w) for w in words])

print("\nLemmatization:")
print([lemmatizer.lemmatize(w) for w in words])


📊 Results

Stemming output may create truncated forms like studies → studi.

Lemmatization output produces meaningful dictionary forms like studies → study.


📝 License

This project is open-source and available under the MIT License.

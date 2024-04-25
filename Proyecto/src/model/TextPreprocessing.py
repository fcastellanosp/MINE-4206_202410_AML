import contractions
import inflect
import nltk
import numpy as np
import pandas as pd
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('all')
#Descargamos únicamente los recursos para idioma inglés
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessing:

    def __init__(self, stopwords=stopwords.words('english')):
        self.stopwords = stopwords

    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub("@\S+", "", word)
            new_word = re.sub(r'[\n\r\t\f\v]', ' ', new_word)
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in self.stopwords:
                new_words.append(word)
        return new_words

    def stem_words(self, words):
        """Stem words in list of tokenized words"""
        # El idioma es inglés, no español
        stemmer = SnowballStemmer('english')
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def stem_and_lemmatize(self, words):
        words = self.stem_words(words)
        words = self.lemmatize_verbs(words)
        return words

    def preproccesing(self, words):
        words = self.to_lowercase(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        return words

    def processing(self, words):
        words = self.preproccesing(words)
        words = [contractions.fix(text) for text in words]
        words = [word_tokenize(text) for text in words]
        words = [self.stem_and_lemmatize(text) for text in words]
        words = ' '.join(map(str, words))
        return words

    """Receives the TF-IDF and PCA as we trained before"""
    def transform(self, x, tfidf, pca):
        x_test_new = self.processing(x)

        x_tfidf = tfidf.transform([x_test_new])
        print(f"TF-IDF shape: {x_tfidf.shape}")

        x_pca = pca.transform(x_tfidf)
        print(f"PCA shape: {x_pca.shape}")

        return x_pca

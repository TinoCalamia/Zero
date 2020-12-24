"""Helper functions for manipulating text."""
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ssl

# Disable ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download("wordnet")


def stem(a):
    """Remove 's' from end if a word."""
    if a.endswith("s"):
        return a[: -len("s")]
    return a


def singularize_words(data):
    """Make singular of plural nouns."""
    wnl = WordNetLemmatizer()
    tokens = [word_tokenize(x)[0] for x in data]
    lemmatized_words = [wnl.lemmatize(token) for token in tokens]

    singular_words = [stem(x) for x in lemmatized_words]

    return singular_words

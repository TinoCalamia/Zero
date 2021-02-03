"""Helper functions for manipulating text."""
import ssl

import inflect

# Disable ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def singularize_words(data):
    """Make singular of plural nouns."""
    p = inflect.engine()

    singular_words = [p.singular_noun(word) for word in data]

    return singular_words

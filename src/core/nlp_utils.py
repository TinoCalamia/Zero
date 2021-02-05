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

    singular_words = [
        p.singular_noun(word) if p.singular_noun(word) is not False else word
        for word in data
    ]
    capitalized_singular_words = [word.capitalize() for word in singular_words]

    return capitalized_singular_words

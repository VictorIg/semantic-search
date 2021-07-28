import string

import nltk
import regex as re


def remove_url(text: str) -> str:
    """
    Removes URL strings from a text.
    :param text: A text string that could contain URLs.
    :return: The text string without URLs.
    """
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_html(text: str) -> str:
    """
    Removes HTML tags from a string. This is needed when our documents come from web scraping.
    :param text: A document, represented as a string.
    :return: The document but without HTML tags.
    """
    return re.sub(r'<.*?>', '', text)


def remove_punctuation(text: str) -> str:
    """
    Removes punctuation from a string. Although not the most clear way to strip punctuation, it provides
    top time efficiency.
    :param text: A document, represented as a string.
    :return: The document but without punctuation.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(text: str) -> str:
    """
    Removes the stop words from a string.
    :param text: A document, represented as a string.
    :return: The document but without stop words.
    """
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    language = 'english'
    language_stopwords = set(stopwords.words(language))

    text_tokens = text.split()
    return " ".join([x for x in text_tokens if x not in language_stopwords])


def remove_spaces(text: str) -> str:
    """
    Removes extra spaces (that may be the result of other preprocessings) from the text.
    :param text: A document, represented as a string.
    :return: The document but without consecutive spaces.
    """
    return re.sub(r'\s+', ' ', text)


def remove_newlines(text: str) -> str:
    """
    Removes new lines from a text.
    :param text:
    :return:
    """
    return text.replace('\n', ' ')


def to_lower(text: str) -> str:
    """
    Transforms the text into lowercase-only text.
    :param text: A document, represented as a string.
    :return: The document, but in lowercase.
    """
    return text.lower()


def preprocess(text: str) -> str:
    """
    Applies multiple preprocessing steps to an input string. The preprocessing steps are contained as method references
    in an array, and the preprocessing is done by iteratively calling the steps, in a chaining fashion. By default,
    all the methods to be chained only take one non-default parameter, the text.
    """
    steps = [remove_url, remove_html, remove_stop_words, remove_punctuation, remove_newlines, remove_spaces]

    for step in steps:
        text = step(text)

    return text

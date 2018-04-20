from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import string
import re

links = re.compile(
    '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})',
    re.IGNORECASE)

punctuation = re.compile(
    r'[^\w\s]'
)

time = re.compile(
    r'(1[012]|[1-9]):[0-5][0-9](\\s)?(?i)(am|pm)'
)


def parse_html_plus_urls(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = links.sub(' ', text)
    return text


def parse_stop_words_punctuation(text):
    text = punctuation.sub(' ', text)
    text = text.strip()
    stop = stopwords.words('english')
    # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
    text = ' '.join([word for word in text.split() if word not in stop])
    return text


def parse_time(text):
    return time.sub(' ', text)


def parse_numbers(text):
    digits = [s for s in text.split() if s.isdigit()]
    text = ' '.join([word for word in text.split() if word not in digits])
    return text


def remove_words_length(text):
    text = ' '.join([word for word in text.split() if len(word) > 2 and len(word) < 16])
    return text


def parse_text(text):
    text = text.lower()
    text = parse_html_plus_urls(text)
    text = parse_time(text)
    text = parse_stop_words_punctuation(text)
    text = parse_numbers(text)
    text = remove_words_length(text)
    return text

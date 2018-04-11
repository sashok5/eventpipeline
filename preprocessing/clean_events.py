
from bs4 import BeautifulSoup
import unicodedata
import sys
from nltk.corpus import stopwords
import pandas as pd
import string
import os


def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')

    text = soup.get_text()

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    text = text.strip()

    stop = stopwords.words('english')

    # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
    text = ' '.join([word for word in text.split() if word not in stop])

    text = text.lower()

    return text


def clean_events_texts(file_in, file_out, reset=False):
    if not os.path.exists(file_in) or reset is True:
        events = pd.read_csv(file_in)
        events = events[['event_id', 'event_name', 'description']]
        events = events.dropna(how='any')
        events['description'] = events['description'].apply(clean_text)
        events['event_name'] = events['event_name'].apply(clean_text)
        events.to_csv(file_out, encoding='utf-8', index=False)

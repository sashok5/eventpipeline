import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
# import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt
import random


matplotlib.interactive(False)
nltk.download('stopwords')
nltk.download('punkt')

# from sklearn.feature_extraction.text import TfidfVectorizer

users_file = 'tables_as_csv/users.csv'
future_events_file = 'tables_as_csv/events.csv'
past_events_file = 'tables_as_csv/past_events.csv'


def stop_word_filter(content):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(content)
    filtered_content = [word for word in words if word not in stop_words]
    return filtered_content


def bar_plot(results_dict):
    # getting common keys from both dictionaries
    key_values = []
    for each_key in results_dict[1].keys():
        if each_key in results_dict[0].keys():
            key_values.append(each_key)

    # plotting the graph here
    values1 = [results_dict[0][each_key] for each_key in key_values]
    values2 = [results_dict[1][each_key] for each_key in key_values]
    ind = np.arange(len(key_values))

    width = 0.35  # set bar_width

    # plt.bar(ind, values1, color='r', width=width, edgecolor='white', label='suggestions')
    # plt.bar(ind + width, values2, color='y', width=width, edgecolor='white', label='attended in past')
    #
    # plt.xlabel('words', fontweight='bold')
    # plt.xticks(ind + width / 2, key_values, rotation=60)
    # plt.ylabel('frequency')
    # plt.legend(loc='best')
    # plt.show()


class Events:
    def __init__(self, events):
        self.events = events
        self.user_df = pd.read_csv(users_file)

    def map_events(self, row):
        try:
            interested_categories = row['interested_categories'].split(',')
        except Exception as e:
            interested_categories = []
        events = []
        if len(interested_categories) == 0:
            # if user has no interests, we will load all the events to his suggestion
            # events = list(self.events['event_description'])
            return events
        for each_category in interested_categories:
            if 'category' in self.events.columns:
                df = self.events.loc[self.events['category'] == each_category]  # grouping by categories
                sub_events = list(df['event_description'])
                for each_event in sub_events:
                    events.append(each_event)
            else:
                sub_events = list(self.events['event_description'])
                for each_event in sub_events:
                    if each_event.find(each_category.lower()) > 1:
                        events.append(each_event)
        return events

    def build_user(self):
        self.user_df['suggested_events'] = self.user_df.apply(self.map_events, axis=1)
        self.user_df['total_suggested_events'] = self.user_df.apply(lambda x: len(x['suggested_events']), axis=1)
        self.user_df['no_of_interests'] = self.user_df.apply(lambda x: len(x['interests'].split(',')), axis=1)
        return self.user_df


class PresentAndPast:
    def __init__(self):
        self.future_events = pd.read_csv(future_events_file, encoding='utf-8')
        self.past_events = pd.read_csv(past_events_file, encoding="iso-8859-1")

    # building user tables (Past and Present here)
    def user_df(self, identifier="None"):
        if identifier.lower() not in ["future", "past"]:
            return
        if identifier.lower() == "future":
            events = self.future_events
        elif identifier.lower() == "past":
            events = self.__set_past_events()
        df = Events(events).build_user()
        return df

    def __set_past_events(self):
        past_events = self.past_events
        past_events.columns = ['event_description', 'event_name', 'group']
        return past_events

    @staticmethod
    def map_attended_events(row):
        if row['suggested_events'] is None:
            return
        suggested_events = list(row['suggested_events'])
        if len(suggested_events) == 0:
            return
        elif len(suggested_events) >= 400:
            # lets assume, user attended 10 percent of events if suggested are more than 400
            percent = 10
        elif 200 <= len(suggested_events) <= 400:
            # lets assume, user attended 8 percent of events if suggested are more than 200 and less than 400
            percent = 8
        elif 100 <= len(suggested_events) <= 200:
            percent = 15
        elif 1 <= len(suggested_events) <= 100:
            percent = 18
        attended_count = int((len(suggested_events) / 100) * percent)
        events_attended = set(random.choice(suggested_events) for each_item in range(0, attended_count))
        return list(events_attended)

    def run(self):
        future_df = self.user_df("future")
        past_df = self.user_df("past")
        past_df['attended_events'] = past_df.apply(self.map_attended_events, axis=1)
        return [future_df, past_df]


class TextAnalyzer:
    def __init__(self, events, identifier=None):
        self.user_df = events
        if identifier is not None:
            self.identifier = identifier

    @staticmethod
    def interest_counter(interests_list, contents_list):
        bow = interests_list
        event_count = 0
        for each_content in contents_list:
            event_count = event_count + 1
            for each_interest in bow:
                words = stop_word_filter(each_content)
                counts = Counter(words)
                interest_count = counts[each_interest]
                if interest_count > 0:
                    yield {event_count: {each_interest: interest_count}}

    def analyzer(self, user):
        # todo: Not required as of now. ==> Future use.
        users = list(self.user_df['user'])
        suggested_events = list(self.user_df['suggested_events'])
        interests = list(self.user_df['interests'])
        if user in users:
            idx = users.index(user)
            suggested_events_list = suggested_events[idx]
            user_interests = [interest.lower() for interest in list(set(interests[idx].split(',')))]
            interest_meter = [interest_count for interest_count in
                              self.interest_counter(user_interests, suggested_events_list)]
            return len(interest_meter)

    def vector_mapping(self, user):
        users = list(self.user_df['user'])
        if self.identifier == "past_events":
            suggested_events = list(self.user_df['attended_events'])
        else:
            suggested_events = list(self.user_df['suggested_events'])
        interests = list(self.user_df['interests'])
        if user in users:
            idx = users.index(user)
            suggested_events_list = suggested_events[idx]
            user_interests = [interest.lower() for interest in list(set(interests[idx].split(',')))]
            vectorizer = CountVectorizer()
            # create Bag of words
            bag_of_words = vectorizer.fit_transform(suggested_events_list)  # bag of words created.
            bag_of_vocabulary = vectorizer.vocabulary_
            word_name = []
            word_count = []
            for each_interest in user_interests:
                try:
                    word_count.append(str(bag_of_vocabulary[each_interest]))
                    word_name.append(each_interest)
                except KeyError:
                    continue
            return dict(zip(word_name, word_count))


def user_frequency(user_name):
    user_name = str(user_name)
    all_events = PresentAndPast().run()
    event_dict = dict(zip(['future_events', 'past_events'], all_events))
    results = []
    for event_identifier, each_event_group in event_dict.items():
        results.append(TextAnalyzer(each_event_group, identifier=event_identifier).vector_mapping(user_name))
    return results

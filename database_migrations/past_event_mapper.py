import pandas as pd
from random import shuffle
import math


class PastEventMapper:

    def __init__(self):
        groups_file = '../formatted_tables/groups_formatted.csv'
        self.groups_file = pd.read_csv(groups_file, encoding='latin-1')

        # filtering groups with only events here
        self.groups_with_events = self.groups_file.loc[self.groups_file['events'] != '[]']
        # getting users with groups
        users_file = '../formatted_tables/users_formatted.csv'
        self.user_df = pd.read_csv(users_file, encoding='latin-1')

    @staticmethod
    def select_random_events(events_list, percent=10):
        if events_list is None:
            return []
        else:
            # shuffling events_list
            shuffle(events_list)
            slice_length = math.ceil(len(events_list) / percent)
            events_list = events_list[:slice_length]
            return events_list

    def map_events_user(self, row):
        user_vs_events = []
        groups = eval(row['groups'])
        if groups is None:
            return user_vs_events
        for each_group_id in groups:
            events = self.groups_with_events.loc[self.groups_with_events['group_id'] == each_group_id]['events']
            if events is None:
                continue
            elif len(events) > 0:
                events = eval(events.tolist()[0])
                events_list = self.select_random_events(events)
                user_vs_events = user_vs_events + events_list
        return user_vs_events

    def map_category_user(self, row):
        categories = []
        groups = eval(row['groups'])
        if groups is None:
            return categories
        for each_group_id in groups:
            category = self.groups_file.loc[self.groups_file['group_id'] == each_group_id]['category.name']
            if str(category).strip().isdigit():
                continue
            categories.append(str(category))
        return list(set(categories))

    def event_mapper(self):
        self.user_df['categories'] = self.user_df.apply(self.map_category_user, axis=1)
        self.user_df['events'] = self.user_df.apply(self.map_events_user, axis=1)
        self.user_df = self.user_df[self.user_df.astype(str)['events'] != '[]']
        self.user_df.to_csv('../formatted_tables/user_vs_events.csv')
        return

import pandas as pd
from random import shuffle
import math

groups_file = '../formatted_tables/groups_formatted.csv'
groups_with_events = pd.read_csv(groups_file, encoding='latin-1')

# filtering groups with only events here
groups_with_events = groups_with_events.loc[groups_with_events['events'] != '[]']


# getting users with groups
users_file = '../formatted_tables/users_formatted.csv'
user_df = pd.read_csv(users_file, encoding='latin-1')


def select_random_events(events_list, percent=10):
    if events_list is None:
        return []
    else:
        # shuffling events_list
        shuffle(events_list)
        slice_length = math.ceil(len(events_list) / percent)
        events_list = events_list[:slice_length]
        return events_list


def map_events_to_user(row):
    user_vs_events = []
    groups = eval(row['groups'])
    if groups is None:
        return user_vs_events
    for each_group_id in groups:
        events = groups_with_events.loc[groups_with_events['group_id'] == each_group_id]['events']
        if events is None:
            continue
        elif len(events) > 0:
            events = eval(events.tolist()[0])
            events_list = select_random_events(events)
            user_vs_events = user_vs_events + events_list
    return user_vs_events


user_df['events'] = user_df.apply(map_events_to_user, axis=1)
user_df.to_csv('../formatted_tables/user_vs_events.csv')

import pandas as pd
import numpy as np


users_file = '../new_tables/members.csv'
# past_events_file = '../new_tables/past_events.csv'
# future_events_file = '../new_tables/events.csv'


users = pd.read_csv(users_file, encoding='latin-1')

""" Index(['member_id', 'bio', 'city', 'country', 'hometown', 'joined', 'lat',
       'link', 'lon', 'member_name', 'state', 'member_status', 'visited',
       'group_id'],
      dtype='object' """


# Breaking down the df to only important features of user
user_df = users[['member_id', 'link', 'member_name', 'state']]
user_df = user_df.drop_duplicates(['member_id', 'link', 'member_name', 'state'])
user_df = user_df[:3000]        # taking only 3000 users for now


def user_groups(row):
    df = users.loc[users['member_id'] == row['member_id']][['group_id']]
    groups = df['group_id'].tolist()
    print(row['member_id'], groups)
    return groups


user_df['groups'] = user_df.apply(user_groups, axis=1)
user_df.to_csv("../formatted_tables/users_formatted.csv")

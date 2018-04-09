import pandas as pd


groups_file = "../new_tables/groups.csv"
groups_df = pd.read_csv(groups_file, encoding='latin-1')

past_events_file = '../new_tables/events.csv'
events_df = pd.read_csv(past_events_file, encoding='latin-1')


""" Index(['group_id', 'category_id', 'category.name', 'category.shortname',
       'city_id', 'city', 'country', 'created', 'description',
       'group_photo.base_url', 'group_photo.highres_link',
       'group_photo.photo_id', 'group_photo.photo_link',
       'group_photo.thumb_link', 'group_photo.type', 'join_mode', 'lat',
       'link', 'lon', 'members', 'group_name', 'organizer.member_id',
       'organizer.name', 'organizer.photo.base_url',
       'organizer.photo.highres_link', 'organizer.photo.photo_id',
       'organizer.photo.photo_link', 'organizer.photo.thumb_link',
       'organizer.photo.type', 'rating', 'state', 'timezone', 'urlname',
       'utc_offset', 'visibility', 'who'],
      dtype='object') """


groups_df = groups_df[['group_id', 'category_id', 'category.name',
                       'category.shortname', 'city', 'country', 'link', 'group_name']]


def events_id(row):
    df = events_df.loc[events_df['group_id'] == row['group_id']][['event_id']]
    events = df['event_id'].tolist()
    return events


groups_df['events'] = groups_df.apply(events_id, axis=1)
groups_df.to_csv('../formatted_tables/groups_formatted.csv')

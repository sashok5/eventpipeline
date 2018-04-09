import pandas as pd


users_file = '../tables_as_csv/users.csv'
past_events_file = '../tables_as_csv/past_events.csv'
future_events_file = '../tables_as_csv/events.csv'


def apply_email(row):
    try:
        email = str(row['user']) + '@testmail.com'
        email = email.replace(' ', '_')
    except ValueError:
        email = ''
    return email


def format_user(userdf):
    userdf['email'] = userdf.apply(apply_email, axis=1)


def format_events(eventsdf):
    pass


# user_df = pd.read_csv(users_file)
# format_user(user_df)
# print(user_df.head())

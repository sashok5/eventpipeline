import pandas as pd


class CategoryMapper:

    def __init__(self):
        self.user_df = pd.read_csv('../formatted_tables/user_vs_events.csv')
        self.categories = pd.read_csv('../formatted_tables/categories.csv')

    def category_mapper(self):
        pass

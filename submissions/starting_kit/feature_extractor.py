import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'),
                            compression='zip', low_memory=False)
        # obtain features from award
        award['Name_processed'] = award['incumbent_name'].str.lower()
        award['Name_processed'] = \
            award['Name_processed'].str.replace(r'[^\w]', '')
        award_features = \
            award.groupby(['Name_processed'])['amount'].agg(['count', 'sum'])

        def zipcodes(X):
            zipcode_nums = pd.to_numeric(X['Zipcode'], errors='coerce')
            return zipcode_nums.values[:, np.newaxis]
        zipcode_transformer = FunctionTransformer(zipcodes, validate=False)

        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))])

        def process_date(X):
            date = pd.to_datetime(X['Fiscal_year_end_date'], format='%Y-%m-%d')
            return np.c_[date.dt.year, date.dt.month, date.dt.day]
        date_transformer = FunctionTransformer(process_date, validate=False)

        def process_APE(X):
            APE = X['Activity_code (APE)'].str[:2]
            return pd.to_numeric(APE).values[:, np.newaxis]
        APE_transformer = FunctionTransformer(process_APE, validate=False)

        def merge_naive(X):
            X['Name'] = X['Name'].str.lower()
            X['Name'] = X['Name'].str.replace(r'[^\w]', '')
            df = pd.merge(X, award_features, left_on='Name',
                          right_on='Name_processed', how='left')
            return df[['count', 'sum']]
        merge_transformer = FunctionTransformer(merge_naive, validate=False)

        num_cols = ['Legal_ID', 'Headcount',
                    'Fiscal_year_duration_in_months', 'Year']
        zipcode_col = ['Zipcode']
        date_cols = ['Fiscal_year_end_date']
        APE_col = ['Activity_code (APE)']
        merge_col = ['Name']
        drop_cols = ['Address', 'City']

        preprocessor = ColumnTransformer(
            transformers=[
                ('zipcode', make_pipeline(zipcode_transformer,
                 SimpleImputer(strategy='median')), zipcode_col),
                ('num', numeric_transformer, num_cols),
                ('date', make_pipeline(date_transformer,
                 SimpleImputer(strategy='median')), date_cols),
                ('APE', make_pipeline(APE_transformer,
                 SimpleImputer(strategy='median')), APE_col),
                ('merge', make_pipeline(merge_transformer,
                 SimpleImputer(strategy='median')), merge_col),
                ('drop cols', 'drop', drop_cols),
                ])

        self.preprocessor = preprocessor
        self.preprocessor.fit(X_df, y_array)
        return self

    def transform(self, X_df):
        return self.preprocessor.transform(X_df)

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
        pass

    def transform(self, X_df):
        X_encoded = X_df

        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv'),
                            low_memory=False)
        # obtain features from award
        award['Name_processed'] = award['incumbent_name'].str.lower()
        award['Name_processed'] = award['Name_processed'].str.replace('[^\w]','')
        award_features = award.groupby(['Name_processed'])['amount'].agg(['count','sum'])

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
            X['Name'] = X['Name'].str.replace('[^\w]','')
            df = pd.merge(X, award_features, left_on='Name', 
                          right_on='Name_processed', how='left')
            return df[['count','sum']]
        merge_transformer = FunctionTransformer(merge_naive, validate=False)

        num_cols = X_encoded.select_dtypes([np.number]).columns
        date_cols = ['Fiscal_year_end_date']
        APE_col = ['Activity_code (APE)']
        merge_col = ['Name']
        drop_cols = ['Address', 'City']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('date', make_pipeline(date_transformer, SimpleImputer(strategy='median')), date_cols),
                ('APE', make_pipeline(APE_transformer, SimpleImputer(strategy='median')), APE_col),
                ('merge', make_pipeline(merge_transformer, SimpleImputer(strategy='median')), merge_col),
                ('drop cols', 'drop', drop_cols),
                ])

        X_array = preprocessor.fit_transform(X_encoded)
        return X_array
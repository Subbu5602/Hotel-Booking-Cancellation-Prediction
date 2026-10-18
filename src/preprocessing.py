import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline   
import os

file_path = os.path.join(os.path.dirname(__file__),"../data/hotel_bookings.csv")

def load_data(file_path=file_path):
   df = pd.read_csv(file_path)
   X = df.drop(["is_canceled","reservation_status","reservation_status_date","agent","company","country"], axis=1)
   y = df["is_canceled"]
   return X, y

def build_preprocessor(X):
   numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
   categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
   num_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="mean"))])
   cat_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])

   preprocessor = ColumnTransformer(transformers=[("number",num_pipeline,numeric_features),("category",cat_pipeline,categorical_features)])

   return preprocessor


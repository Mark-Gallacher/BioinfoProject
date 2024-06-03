from Pipeline import Pipeline
from FeatureElimination import FeatureElimination
from Metrics import Metric

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

import os

threads = os.cpu_count()

if threads is None:
    threads = 1

print()
print(f"There appear to be {threads} threads available!!")
print()

img_dir = "../png/feature_selection"
out_dir = "../data/feature_selection"

num_folds = 10

#### Loading in the Data ####
_raw_data = pd.read_csv("../data/SubTypeData.csv")
raw_data = _raw_data.copy().drop(["DiseaseSubtypeFull", "PseudoID"], axis = 1)

### Explicitly controlling the folds, so they are the same across all models
folds = StratifiedKFold(n_splits = num_folds, random_state = 1, shuffle = True)

### Creating the test and train sets
sss = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 1)

for train_i, test_i in sss.split(raw_data,  raw_data["DiseaseSubtype"]):
    train_set = raw_data.loc[train_i]
    test_set = raw_data.loc[test_i]

_unscaled_train_data = train_set.drop("DiseaseSubtype", axis = 1)
train_labels = train_set["DiseaseSubtype"].copy()

### Scale the Features - Only looking at the training data for now, not the testing set
scaler = StandardScaler()
columns = _unscaled_train_data.columns
train_data = scaler.fit_transform(_unscaled_train_data[columns])

### Feature Extraction

lg_rfe = FeatureElimination(
            model_name = "LogisticRegressionRFE", 
            model_code = "LG", 
            model = LogisticRegression, 
            n_jobs = threads, 
            folds = folds, 
            out_dir = out_dir,
            max_iter = 5000, 
            solver = "saga")

rf_rfe = FeatureElimination(
            model_name = "RandomForestRFE", 
            model_code = "RF",
            model = RandomForestClassifier,
            folds = folds, 
            n_jobs = threads, 
            out_dir = out_dir)

gb_rfe = FeatureElimination(
            model_name = "GradientBoostingRFE", 
            model_code = "GB", 
            model = GradientBoostingClassifier, 
            folds = folds, 
            n_jobs = threads, 
            out_dir = out_dir)

#### Metrics ####
base_metric = "balanced_accuracy"


for model in [lg_rfe, rf_rfe, gb_rfe]:

    pipeline = Pipeline(model = model, metric_spec = base_metric)

    df = pipeline.generate_metric_dataframe(X = train_data, y = train_labels)

    pipeline.save_as_csv(df, out_dir)

    pipeline.model_class.extract_best_features(raw_df = _raw_data)



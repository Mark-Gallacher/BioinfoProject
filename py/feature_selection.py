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

num_folds = 10

#### Loading in the Data ####
_raw_data = pd.read_csv("../data/SubTypeData.csv")
raw_data = _raw_data.drop(["DiseaseSubtypeFull", "PseudoID"], axis = 1)

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
lg_rfe = RFECV(estimator = LogisticRegression(max_iter = 5000, solver = "saga"), 
            cv = folds, 
            step = 1,
            scoring = "balanced_accuracy",
            min_features_to_select = 1, 
            n_jobs = threads)

# nb_rfe = RFECV(estimator = GaussianNB(), 
#             cv = folds, 
#             step = 1,
#             scoring = "balanced_accuracy",
#             min_features_to_select = 1, 
#             n_jobs = threads)

rf_rfe = RFECV(estimator = RandomForestClassifier(), 
            cv = folds, 
            step = 1,
            scoring = "balanced_accuracy",
            min_features_to_select = 1, 
            n_jobs = threads)

gb_rfe = RFECV(estimator = GradientBoostingClassifier(min_samples_split = 10), 
            cv = folds, 
            step = 1,
            scoring = "balanced_accuracy",
            min_features_to_select = 1, 
            n_jobs = threads)

rf_rfe.set_output(transform = "pandas")
lg_rfe.set_output(transform = "pandas")
# nb_rfe.set_output(transform = "pandas")
gb_rfe.set_output(transform = "pandas")

rf_new_feautres = rf_rfe.fit_transform(X = train_set, y = train_labels)
lg_new_feautres = lg_rfe.fit_transform(X = train_set, y = train_labels)
# nb_new_feautres = nb_rfe.fit_transform(X = train_set, y = train_labels)
gb_new_feautres = gb_rfe.fit_transform(X = train_set, y = train_labels)

rf_new_feautres.to_csv("../data/rf_new_features.csv")
lg_new_feautres.to_csv("../data/lg_new_features.csv")
# nb_new_feautres.to_csv("../data/lg_new_features.csv")
gb_new_feautres.to_csv("../data/gb_new_features.csv")



cv_results = pd.DataFrame(rf_rfe.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.savefig(f"{img_dir}/RF_RFE.png")

cv_results = pd.DataFrame(lg_rfe.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.savefig(f"{img_dir}/LG_RFE.png")

# cv_results = pd.DataFrame(nb_rfe.cv_results_)
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Mean test accuracy")
# plt.errorbar(
#     x=cv_results["n_features"],
#     y=cv_results["mean_test_score"],
#     yerr=cv_results["std_test_score"],
# )
# plt.title("Recursive Feature Elimination \nwith correlated features")
# plt.savefig("NB_RFE.png")

cv_results = pd.DataFrame(gb_rfe.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.savefig(f"{img_dir}/GB_RFE.png")

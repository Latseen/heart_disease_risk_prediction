# train.py
# Chet Russell
# Training file to train XGBoost model on values given from dataset

import matplotlib.pyplot as pyplot
import pandas as pd
import xgboost as xgb
from numpy import absolute
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance, XGBModel
from xgboost import plot_tree

classification = True
return_graphs = True

data = pd.read_csv("heart_disease_risk_dataset_earlymed.csv")
xgb.set_config(verbosity=2)

# getting data into xgboost format
dataset = data.values

# split data into input and output columns
X, y = dataset[:, :-1], dataset[:, -1]

# scaling data
# NOTE: scaling data is not necessary when utilizing boosted trees. See here: https://github.com/dmlc/xgboost/issues/357
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(X)
model = XGBModel
scoring = ""

if classification:
    # define model
    model = xgb.XGBClassifier(
        tree_method="hist",
        objective="reg:logistic",
        eval_metric=["logloss", "error"],
        enable_categorical=True,
        # device='gpu'
    )
    scoring = "accuracy"
else:
    model = xgb.XGBRegressor()
    scoring = "neg_mean_absolute_error"

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)
# evaluate model
scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring=scoring)

if classification:
    print("Accuracy: ", "{:.3f}".format(scores.mean()))
else:
    scores = absolute(scores)
    print(scoring, "{:.3f}".format(scores.mean()))

# train and save model
model.fit(X, y)
if classification:
    model.save_model("model_classification.json")
else:
    model.save_model("model_regression.json")

if return_graphs:
    # feature importance
    # getting feature names
    data_top = data.head(0)
    feature_names = list(data_top)

    feature_names.pop()
    model.get_booster().feature_names = feature_names

    # plot feature importance
    plot_importance(model)
    pyplot.show()

    # plot single tree
    plot_tree(model)
    pyplot.show()
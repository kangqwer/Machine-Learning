
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import lightgbm as lgb

features_file = r'E:\TCGA\DeepMicroCancer-master\DeepMicroCancer-master\Data\data\blood_snm.csv'
labels_file = r'E:\TCGA\DeepMicroCancer-master\DeepMicroCancer-master\Data\data\blood_meta.csv'
test_size = 0.2

# Load Data
X = pd.read_csv(features_file, index_col = 0)
y = pd.read_csv(labels_file, index_col = 0)['disease_type']

y = y[~y.isin(['Kidney Chromophobe', 'Kidney Renal Clear Cell Carcinoma', 
               'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma'])]
X = X.loc[y.index]    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, 
                                                    random_state = 0)

#Build Model
le = LabelEncoder()
labels = le.fit_transform(y_train)

lr_model = LogisticRegression()

rf_model = RandomForestClassifier(n_estimators=30, criterion='entropy', 
                                  random_state=13, n_jobs=-1)

lgbm_model = lgb.LGBMClassifier(n_estimators=20,learning_rate=0.1,verbosity=-1,
                                random_state=13,n_jobs=-1,)

base_models = [('RandomForest', rf_model), ('LGBM', lgbm_model)]

clf = StackingClassifier(estimators=base_models,final_estimator=lr_model)
clf.fit(X_train, labels)

#Predict
features = X_train.columns.values
include_feature = set(features).intersection(set(X_test.columns))
fill0_feature = set(features) - set(X_test.columns)
include_feature_list = list(include_feature)
fill0_feature_list = list(fill0_feature)
X_test = pd.concat([X_test.loc[:, include_feature_list],
                    pd.DataFrame(np.zeros((X_test.shape[0], len(fill0_feature_list))),
                                 index=X_test.index, columns=fill0_feature_list)], axis=1)
X_test = X_test.loc[:, features]

y_pre = clf.predict(X_test)
y_pre = le.inverse_transform(y_pre)
y_pre_proba = clf.predict_proba(X_test)
result_df = pd.DataFrame(y_pre_proba, index = X_test.index, columns = le.classes_)
result_df['prediction'] = y_pre

skplt.metrics.plot_roc(y_test, y_pre_proba)
plt.title("ROC Curve")
plt.show()
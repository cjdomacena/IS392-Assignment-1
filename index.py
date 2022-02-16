import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

########################################
### Q1
########################################
# With Categories
train_data = pd.read_csv('news-train-1.csv')
# Test Data
test_data = pd.read_csv('news-test.csv')

with open('dictionary.txt') as f:
    dic = f.read().splitlines()

train_data_text = train_data['Text']

vectorizer = TfidfVectorizer(
    stop_words='english', vocabulary=dic, analyzer='word', lowercase=True)

train_data_X = vectorizer.fit_transform(train_data_text)
feature_names = vectorizer.get_feature_names_out()
df = pd.DataFrame(train_data_X.toarray(), columns=feature_names)


categories = train_data['Category']
encoder = LabelEncoder()
encoded_cat = encoder.fit_transform(categories)

########################################
### Q2A
########################################
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(
    df, encoded_cat, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
gini = DecisionTreeClassifier(criterion='gini', random_state=0)
gini.fit(X_train, y_train)

gini_train_accuracy = gini.score(X_train, y_train)
gini_test_val_accuracy = gini.score(X_test, y_test)

entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=0)
entropy.fit(X_train, y_train)
entropy_train_accuracy = entropy.score(X_train, y_train)
entropy_test_val_accuracy = entropy.score(X_test, y_test)

data = [1 - gini_train_accuracy, 1 - entropy_train_accuracy,
        1 - gini_test_val_accuracy, 1 - entropy_test_val_accuracy]

# https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/
def addlabels(x):
    for i in range(len(x)):
        plt.text(i,x[i], np.format_float_positional(x[i],precision=3), ha='center')

X = np.arange(4)
plt.figure(figsize=(12,8))
plt.bar(X, data, width = .8)
plt.xticks(X,labels=['gini_train_accuracy', 'entropy_train_accuracy', 'gini_train_validation', 'gini_test_validation'])
addlabels(data)
plt.title('training error and validation error w.r.t.')
plt.savefig('hw_01_Q2A.png')
plt.show()

# ########################################
# ### Q2B
# ########################################
param_dist = {
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': range(1, 18),
    'max_features': [150, 500, 750, 1000]
}

dtc = DecisionTreeClassifier()

grid_search = GridSearchCV(dtc, param_dist,cv=5, return_train_score=True)
grid_search.fit(df, encoded_cat)
res = pd.DataFrame(grid_search.cv_results_['params'])
res_criterion = res['criterion']
res['mean_train_score'] = grid_search.cv_results_['mean_train_score']
res['mean_test_score'] = grid_search.cv_results_['mean_test_score']
gini_res = []

def plot_tuned_data(criterion, res, res_criterion):
  plt.figure(figsize=(12,8))
  for i in param_dist['max_features']:
      gini_res = res[(res_criterion == criterion) & (res['max_features'] == i)]
      plt.plot(gini_res['min_samples_leaf'], 1 - gini_res['mean_train_score'], label=f'{criterion} - Training')
      plt.plot(gini_res['min_samples_leaf'], 1 -
              gini_res['mean_test_score'], linestyle='dashed', label=f'{criterion} - Validation')
  plt.xlabel('min_samples_leaf')
  plt.ylabel('Average Error')
  plt.legend(title='max_features')
  file = 'hw_01_Q2B-{f}'.format(f = criterion)
  plt.savefig(file)
  plt.show()


plot_tuned_data('gini', res, res_criterion)
plot_tuned_data('entropy', res, res_criterion)

# ########################################
# ### Q3
# ########################################

n_estimators = range(100, 1001, 25)
rf_param = {
    'n_estimators': n_estimators
}
random_forest = GridSearchCV(RandomForestClassifier(
), rf_param, cv=5, verbose=10)
random_forest.fit(df.to_numpy(), encoded_cat)

# PLOT DATA FOR Q3
plt.plot(n_estimators, random_forest.cv_results_[
         'mean_test_score'], marker='.', label='Avg. Accuracy')
plt.title('Random Forest W.R.T Avg. Accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Average accuracy')
plt.legend()
plt.savefig('hw01_rf_avg_acc.png')
plt.show()

plt.plot(n_estimators, random_forest.cv_results_['std_test_score'],
         label='Std. Deviation', marker='.')
plt.title('Random Forest W.R.T Std. Deviation')
plt.xlabel('n_estimators')
plt.ylabel('Standard Deviation')
plt.legend()
plt.savefig('hw01_rf_std_dev.png')
plt.show()

# # DISPLAY TABLE FOR Q3
rf_res_dic = {
    'n_estimators': n_estimators,
    'Avg. Accuracy': random_forest.cv_results_[
        'mean_test_score'],
    'Std. Deviation': random_forest.cv_results_['std_test_score']
}
rf_df = pd.DataFrame(rf_res_dic)
rf_df.to_html('hw1_q3.html')
print('\n ###################################################### \n')
print('Random forest best score: ', random_forest.best_score_)
print('Random forest best estimator: ', random_forest.best_estimator_)
print('\n ###################################################### \n')


# ########################################
# ### Q4
# ########################################
eta = np.arange(.01, 1, .05)
clf_xgb = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic',
                       use_label_encoder=False), {'learning_rate': eta}, verbose=10, cv=5)
clf_xgb.fit(df.to_numpy(), encoded_cat, eval_metric='rmse')

# DISPLAY TABLE FOR Q4
clf_xgb_res_dic = {
    'Avg. Accuracy': clf_xgb.cv_results_['mean_test_score'],
    'Std.Deviation': clf_xgb.cv_results_['std_test_score'],
    'eta': eta
}


# # PLOT DATA FOR Q4
plt.plot(eta, clf_xgb.cv_results_[
         'mean_test_score'], marker='.', label='Avg. Accuracy')

plt.title('XG Boosting W.R.T Avg. Accuracy')
plt.xlabel('learning_rate')
plt.ylabel('Average accuracy')
plt.legend()
plt.savefig('hw01_xgb_avg_acc.png')
plt.show()

plt.plot(eta, clf_xgb.cv_results_['std_test_score'],
         label='Std. Deviation', marker='.', linestyle='dashed')
plt.title('XG Boosting W.R.T Std. Deviation')
plt.xlabel('learning_rate')
plt.ylabel('Standard Deviation')
plt.legend()
plt.savefig('hw01_xgb_std_dev.png')
plt.show()
xgb_df = pd.DataFrame(clf_xgb_res_dic)
xgb_df.to_html('hw1_xgb.html')

print('\n ###################################################### \n')
print('XG Boost best score: ', clf_xgb.best_score_)
print('XG Boost  best learning rate: ', clf_xgb.best_params_)
print('\n ###################################################### \n')

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedS
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from scipy.stats import loguniform, uniform

wine = load_wine(return_X_y=False, as_frame=True)
# Checks the bunch objects available throgh keys
print(wine.keys())


display(wine.data)

wine.target.value_counts()

wine.data.info()

wine.data.describe().transpose()

# Splits the data into train and test set with stratification on wine class
X_train, X_test, y_train, y_test = train_test_split(
 wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target)
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

pd.DataFrame({
 "Train set": y_train.value_counts()/y_train.size,
 "Test set": y_test.value_counts()/y_test.size
 })


std_scaler = StandardScaler()
# Fits the standard scaler on train set
std_scaler.fit(X_train)
# Transforms the train set
X_train_transformed = std_scaler.transform(X_train)

X_test_transformed = std_scaler.transform(X_test)


lin_svc = LinearSVC(random_state=42)
# Fits the model on train set
lin_svc.fit(X_train_transformed, y_train)

lin_svc_cv = cross_val_score(LinearSVC(random_state=42), X_train_transformed, y_train)
print("Linear SVC CV Score: {:.3f}".format(lin_svc_cv.mean()))

lin_svc_test_predictions = lin_svc.predict(X_test_transformed)
print("Linear SVC Test Score: {:.3f}".format(accuracy_score(y_test, lin_svc_test_predictions)))

svm_clf = SVC(random_state=42)
# Fits the model on train set
svm_clf.fit(X_train_transformed, y_train)

svm_clf_train_predictions = svm_clf.predict(X_train_transformed)
print("SVM Classifier Train Score: {:.3f}".format(accuracy_score(y_train, svm_clf_train_predictions)))

svm_clf_cv = cross_val_score(SVC(random_state=42), X_train_transformed, y_train)
print("SVM Classifier CV Score: {:.3f}".format(svm_clf_cv.mean()))

svm_clf.get_params()

param_distributions = {
 "gamma": loguniform(0.001, 0.1),
 "C": uniform(1, 10)
}
# Performs random search for the best values for the mentioned parameters
rnd_search_cv = RandomizedSearchCV(SVC(random_state=42), param_distributions, n_iter=100, cv=5, random_state=42)
rnd_search_cv.fit(X_train_transformed, y_train)

rnd_search_cv.best_score_

rnd_search_cv.best_estimator_.get_params()


accuracy_score(y_test, rnd_search_cv.best_estimator_.predict(X_test_transformed))


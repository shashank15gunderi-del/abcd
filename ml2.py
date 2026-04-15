import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", as_frame=False)

print(mnist.keys())

print(mnist.data.shape)

print(mnist.target.shape)

print(mnist.target[:20])

for idx, image_data in enumerate(mnist.data[:100]):
 plt.subplot(10, 10, idx + 1)
 image = image_data.reshape(28, 28)
 plt.imshow(image, cmap = "binary")
 plt.axis("off")
plt.show()


mnist.target = mnist.target.astype(int)


X_train, X_test, y_train, y_test = train_test_split(
 mnist.data, mnist.target, test_size=10000, random_state=42, stratify=mnist.target

 y_train

 std_scaler = StandardScaler()
# Scales the feature values
X_train_scaled = std_scaler.fit_transform(X_train)

X_test_scaled = std_scaler.transform(X_test)

sgd_clf = SGDClassifier(n_jobs=-1, random_state=42)
# Fits the model on train set
# NOTE: This step may take few minutes to complete
sgd_clf.fit(X_train_scaled, y_train)

predictions_test = sgd_clf.predict(X_test_scaled)
print(predictions_test)

print("Prediction:", predictions_test[0])
print("Actual Label:", y_test[0])

decision_scores = sgd_clf.decision_function([X_test_scaled[0]])
print(decision_scores)

print("Prediction:", sgd_clf.classes_[decision_scores.argmax()])
print("Actual Label:", y_test[0])

print("Prediction Performance (Accuracy) of SGD Classifier: {:.1f}%".format(accuracy_score(y_test, predictions_test) * 100))

cv_predictions = cross_val_predict(
 sgd_clf, X_train_scaled, y_train, cv=5, n_jobs=-1, verbose=3, method="predict")

 ConfusionMatrixDisplay.from_predictions(y_train, cv_predictions)
plt.title("Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_train, cv_predictions, normalize="true",
plt.title("Confusion Matrix [Row-Normalized]")
plt.show()

sample_weight = (cv_predictions != y_train)
ConfusionMatrixDisplay.from_predictions(
 y_train, cv_predictions, sample_weight=sample_weight, normalize="true", values
plt.title("Confusion Matrix [Error-Normalized by Row]")
plt.show()
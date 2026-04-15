import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import root_mean_squared_error


with open("./housing_train_transformed.npy", "rb") as f:
 X_train_transformed = np.load(f)
with open("./housing_test_transformed.npy", "rb") as f:
 X_test_transformed = np.load(f)

 y_train = X_train_transformed[:, -1]
y_test = X_test_transformed[:, -1]
X_train_transformed = X_train_transformed[:, :-1]
X_test_transformed = X_test_transformed[:, :-1]


theta_ne = np.linalg.inv(X_train_transformed.T @ X_train_transformed) @ X_train
"""
The same expression mentioined above can also be represented like the one below
theta = np.linalg.inv(X_train_transformed.T.dot(X_train_transformed)).dot(X_train_transformed.T).dot(y_train)
"""
# Shows θ values for normal equation (ne)
print(theta_ne)

predictions_train_ne = X_train_transformed.dot(theta_ne)
predictions_test_ne = X_test_transformed.dot(theta_ne)

rmse_train_ne = root_mean_squared_error(y_train, predictions_train_ne)
rmse_test_ne = root_mean_squared_error(y_test, predictions_test_ne)
print("Normal Equation-based Linear Regression Performance [RMSE]:\n{:.1f} [Tra set], {:.1f} [Test set]".format(rmse_train_ne, rmse_test_ne))
 rmse_train_ne, rmse_test_ne))


lin_reg = LinearRegression()
# Fits the model
lin_reg.fit(X_train_transformed, y_train)

print("Linear Regression Model Parameters:\n")
print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)


predictions_train_lr = lin_reg.predict(X_train_transformed)
predictions_test_lr = lin_reg.predict(X_test_transformed)


rmse_train_lr = root_mean_squared_error(y_train, predictions_train_lr)
rmse_test_lr = root_mean_squared_error(y_test, predictions_test_lr)
print("SVD-based Linear Regression Performance [RMSE]:\n{:.1f} [Train]\n{:.1f} [Test]".format(rmse_train_lr, rmse_test_lr))
rmse_train_lr = root_mean_squared_error(y_train, predictions_train_lr)


sgd_reg = SGDRegressor(
 penalty=None, # No regularization
 tol=1e-5, # Tolerance for loss drop during last 'n_iter_no_ch
 max_iter=1000, # Maximum number of traning iterations (epochs)
 eta0=0.01, # Learning rate
 n_iter_no_change=100, # Exits if training loss doesn't improve by 'tol' f
 random_state=42)
# Fits the model
sgd_reg.fit(X_train_transformed, y_train)

print("SGD Model Parameters:\n")
print("Intercept:", sgd_reg.intercept_)
print("Coefficients:", sgd_reg.coef_)


predictions_train_sgd = sgd_reg.predict(X_train_transformed)
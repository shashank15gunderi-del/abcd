import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

life_sat_index = pd.read_csv("./../data/lifesat.csv")
display(life_sat_index)

X = life_sat_index[["GDP per capita (USD)"]].values
y = life_sat_index[["Life satisfaction"]].values
life_sat_index.plot(kind='scatter', x="GDP per capita (USD)", y="Life satisf
plt.axis([23500, 62500, 4, 9])
plt.show()


# Initializes a simple linear model
lin_reg = LinearRegression()
# Fits the linear model
lin_reg.fit(X, y)

print("Intercept:", lin_reg.intercept_, ", Slope:", lin_reg.coef_)
print("\nFormated Output: Intercept: {0:.2f}, Slope: {1:.2e}".format(lin_reg.intercept_[0], lin_reg.coef_[0][0]))

life_sat_index.plot(kind='scatter', x="GDP per capita (USD)", y="Life satisf
plt.axis([23500, 62500, 4, 9])
# Extracts slope and intercept
m = lin_reg.coef_[0][0]
c = lin_reg.intercept_[0]
# Plots the line directly with slope and intersect
plt.axline(xy1=(0, c), slope=m, color="blue", label=f'$y={m}x {c:+}$')
# Shows the linear model equation by enabling legend
plt.legend()
# Finally, renders the plot
plt.draw()


X_test = [[37_655.2]] # Consider it as Cyprus' GDP per capita in 2020
y_predictions = lin_reg.predict(X_test)
# Prints the predictions
print(y_predictions)

life_sat_index.plot(kind='scatter', x="GDP per capita (USD)", y="Life satisf
plt.axis([23500, 62500, 4, 9])
plt.axline(xy1=(0, c), slope=m, color="blue", label=f'$y={m}x {c:+}$')
# Now, plots the predictions
plt.scatter(
 X_test,
 y_predictions,
 c=np.array(["red"])
)
plt.show()

# Initializes k-nearest neighbors regression algorithm,
knn_reg = KNeighborsRegressor(n_neighbors = 3)
# and fits the model
knn_reg.fit(X, y)

y_predictions = knn_reg.predict(X_test)
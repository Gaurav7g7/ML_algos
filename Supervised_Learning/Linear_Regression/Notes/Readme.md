We want to predict a target variable y (here: car price) using features x (like age, mileage, trim, etc.):

y = β0 + β1·x1 + β2·x2 + … + βn·xn


y: the actual value (car price in dataset)

ŷ: the predicted value (our model’s guess)

β0: the intercept (baseline value when all features are 0)

β1, β2, …, βn: the coefficients (weights) — how much each feature contributes to price

x1, x2, …, xn: the input features (age, mileage, LE_flag, Hybrid_flag, etc.)

n: number of features

m: number of data points (cars in dataset)

👉 Example (car price):

Price = β0 + β1·Age + β2·Mileage + β3·LE_Flag + β4·Hybrid_Flag

Cost Function (MSE)

We measure how wrong our predictions are using Mean Squared Error:

J(β) = (1/m) Σ (ŷ(i) - y(i))²


J(β): the cost (error of the model)

y(i): the actual price of car i

ŷ(i): the predicted price of car i

m: total number of cars (rows)

👉 Intuition: The smaller J(β), the better our model fits.

Prediction Function

For each car i:

ŷ(i) = β0 + Σ βj·xj(i)


ŷ(i): predicted price for car i

xj(i): value of feature j for car i

βj: weight assigned to feature j

Gradient Descent Updates

We improve the coefficients step by step to minimize J(β):

βj = βj - α · (∂J/∂βj)


βj: the current weight for feature j

α: the learning rate (step size — too big = overshoot, too small = slow)

∂J/∂βj: the gradient (slope of cost function with respect to βj)

Gradient Formula
∂J/∂βj = (2/m) Σ (ŷ(i) - y(i))·xj(i)


Measures how much feature j contributes to the error

If gradient is positive → decrease βj

If gradient is negative → increase βj

Repeat until J(β) is minimized

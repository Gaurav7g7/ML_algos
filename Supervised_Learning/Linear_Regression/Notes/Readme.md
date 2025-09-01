
Linear Regression Model

We want to predict a target variable 
𝑦
y (here: car price) using features 
𝑥
x (like age, mileage, trim, etc.).

𝑦
=
𝛽
0
+
𝛽
1
𝑥
1
+
𝛽
2
𝑥
2
+
⋯
+
𝛽
𝑛
𝑥
𝑛
y=β
0
	​

+β
1
	​

x
1
	​

+β
2
	​

x
2
	​

+⋯+β
n
	​

x
n
	​


𝑦
y: the actual value (car price in dataset).

𝑦
^
y
^
	​

: the predicted value (our model’s guess).

𝛽
0
β
0
	​

: the intercept (baseline value when all features are 0).

𝛽
1
,
𝛽
2
,
…
,
𝛽
𝑛
β
1
	​

,β
2
	​

,…,β
n
	​

: the coefficients (weights) — how much each feature contributes to price.

𝑥
1
,
𝑥
2
,
…
,
𝑥
𝑛
x
1
	​

,x
2
	​

,…,x
n
	​

: the input features (age, mileage, LE_flag, Hybrid_flag, etc.).

𝑛
n: number of features.

𝑚
m: number of data points (cars in dataset).

👉 Example (car price):

Price
=
𝛽
0
+
𝛽
1
⋅
Age
+
𝛽
2
⋅
Mileage
+
𝛽
3
⋅
LE Flag
+
𝛽
4
⋅
Hybrid Flag
Price=β
0
	​

+β
1
	​

⋅Age+β
2
	​

⋅Mileage+β
3
	​

⋅LE Flag+β
4
	​

⋅Hybrid Flag
🔹 Cost Function (MSE)

We measure how wrong our predictions are using Mean Squared Error:

𝐽
(
𝛽
)
=
1
𝑚
∑
𝑖
=
1
𝑚
(
𝑦
^
(
𝑖
)
−
𝑦
(
𝑖
)
)
2
J(β)=
m
1
	​

i=1
∑
m
	​

(
y
^
	​

(i)
−y
(i)
)
2

𝐽
(
𝛽
)
J(β): the cost (error of the model).

𝑦
(
𝑖
)
y
(i)
: the actual price of car 
𝑖
i.

𝑦
^
(
𝑖
)
y
^
	​

(i)
: the predicted price of car 
𝑖
i.

𝑚
m: total number of cars (rows).

👉 Intuition: The smaller 
𝐽
(
𝛽
)
J(β), the better our model fits.

🔹 Prediction Function

For each car 
𝑖
i:

𝑦
^
(
𝑖
)
=
𝛽
0
+
∑
𝑗
=
1
𝑛
𝛽
𝑗
𝑥
𝑗
(
𝑖
)
y
^
	​

(i)
=β
0
	​

+
j=1
∑
n
	​

β
j
	​

x
j
(i)
	​


𝑦
^
(
𝑖
)
y
^
	​

(i)
: predicted price for car 
𝑖
i.

𝑥
𝑗
(
𝑖
)
x
j
(i)
	​

: value of feature 
𝑗
j for car 
𝑖
i.

𝛽
𝑗
β
j
	​

: weight assigned to feature 
𝑗
j.

🔹 Gradient Descent Updates

We improve the coefficients step by step to minimize 
𝐽
(
𝛽
)
J(β).

𝛽
𝑗
=
𝛽
𝑗
−
𝛼
⋅
∂
𝐽
∂
𝛽
𝑗
β
j
	​

=β
j
	​

−α⋅
∂β
j
	​

∂J
	​


𝛽
𝑗
β
j
	​

: the current weight for feature 
𝑗
j.

𝛼
α: the learning rate (step size — too big = overshoot, too small = slow).

∂
𝐽
∂
𝛽
𝑗
∂β
j
	​

∂J
	​

: the gradient (slope of cost function with respect to 
𝛽
𝑗
β
j
	​

).

🔹 Gradient Formula
∂
𝐽
∂
𝛽
𝑗
=
2
𝑚
∑
𝑖
=
1
𝑚
(
𝑦
^
(
𝑖
)
−
𝑦
(
𝑖
)
)
𝑥
𝑗
(
𝑖
)
∂β
j
	​

∂J
	​

=
m
2
	​

i=1
∑
m
	​

(
y
^
	​

(i)
−y
(i)
)x
j
(i)
	​


Measures how much feature 
𝑗
j contributes to the error.

If gradient is positive → decrease 
𝛽
𝑗
β
j
	​

.

If gradient is negative → increase 
𝛽
𝑗
β
j
	​

.

Repeat until 
𝐽
(
𝛽
)
J(β) is minimized.

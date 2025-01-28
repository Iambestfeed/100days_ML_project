# Problem
Implement using NumPy: Initialize weights, compute predictions $y=Xw+b$, calculate MSE loss, update weights via SGD. Goal: Understand gradient descent mechanics.

# Knowledge
## Linear regression
Problem define: Predicting \( y \) based on \( X \).
- **Formula**:  
  \[
  \hat{y} = \theta_0 + \theta_1 X_1 + \theta_2 X_2 + \dots + \theta_n X_n
  \]  
  Where
  - \( \theta_0 \): Bias term (intercept).
  - \( \theta_1, \theta_2, \dots, \theta_n \): Weights for features \( X_1, X_2, \dots, X_n \).
  - \( \hat{y} \): Predicted value.

### 2. **MSE loss (Mean Squared Error)**
- **Formula**:  
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]  
  Where
  - \( n \): Number of data samples.
  - \( y_i \): True value.
  - \( \hat{y}_i \): Predicted value.


### 3. **Stochastic Gradient Descent (SGD)**  
SGD is an optimization algorithm to **update model weights** \( \theta \) by minimizing MSE.  

#### **Steps**:  
1. **Initialization**: Randomly initialize weights \( \theta \).  
2. **Iterate for multiple epochs**:  
   - **Shuffle data** to avoid order bias.  
   - **For each data point** \( (X^{(i)}, y^{(i)}) \):  
     - **Compute prediction**:  
       \[
       \hat{y}^{(i)} = \theta_0 + \theta_1 X_1^{(i)} + \dots + \theta_n X_n^{(i)}
       \]  
     - **Compute error**:  
       \[
       \text{error} = y^{(i)} - \hat{y}^{(i)}
       \]  
     - **Update weights** (using learning rate \( \eta \)):  
       \[
       \theta_j := \theta_j + \eta \cdot \text{error} \cdot X_j^{(i)} \quad \text{(for } j \geq 1 \text{)}
       \]  
       \[
       \theta_0 := \theta_0 + \eta \cdot \text{error} \quad \text{(for bias term)}
       \]   


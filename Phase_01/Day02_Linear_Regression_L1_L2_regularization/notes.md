# Problem
Implement using NumPy: Normalize input data, initialize weights, compute predictions $\hat{y} = Xw + b$, calculate the Mean Squared Error (MSE) loss with L1 and L2 regularization, and update weights via Stochastic Gradient Descent (SGD) with early stopping.  
**Goal**: Understand the mechanics of gradient descent with regularization and early stopping in linear regression.

# Knowledge

## 1. Linear Regression with Regularization
**Definition**: Predict the target variable $y$ from features $X$ while applying regularization to prevent overfitting.

- **Model Formula**:  
  $\displaystyle \hat{y} = \theta_0 + \theta_1 X_1 + \theta_2 X_2 + \dots + \theta_n X_n$  

  Where:  
  - $\theta_0$: Bias term (intercept).  
  - $\theta_1, \theta_2, \dots, \theta_n$: Weights for features $X_1, X_2, \dots, X_n$.  
  - $\hat{y}$: Predicted value.

## 2. Loss Function with Regularization
### **MSE Loss (Mean Squared Error)**
- **Formula**:  
  $\displaystyle \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2$  

  Where:  
  - $n$: Number of data samples.  
  - $y_i$: True target value for sample $i$.  
  - $\hat{y}_i$: Predicted target value for sample $i$.

### **Regularization Terms**
- **L2 Regularization** (Ridge):  
  $\displaystyle \text{L2} = \lambda_2 \sum_{j=1}^{n} \theta_j^2$  

- **L1 Regularization** (Lasso):  
  $\displaystyle \text{L1} = \lambda_1 \sum_{j=1}^{n} |\theta_j|$  

- **Combined Loss**:  
  $\displaystyle \text{Loss} = \text{MSE} + \text{L1} + \text{L2}$

## 3. Stochastic Gradient Descent (SGD) with Early Stopping
### **Steps**:
1. **Initialization**:  
   - Normalize the data (optional but recommended for stability).  
   - Randomly initialize weights $\theta$ (including bias).

2. **For multiple epochs**:
   - **Shuffle data** to avoid order bias.
   - **For each data sample** $(X^{(i)}, y^{(i)})$:
     - **Compute prediction**:  
       $\displaystyle \hat{y}^{(i)} = \theta_0 + \theta_1 X_1^{(i)} + \dots + \theta_n X_n^{(i)}$
     - **Compute error**:  
       $\displaystyle \text{error}^{(i)} = y^{(i)} - \hat{y}^{(i)}$
     - **Compute gradients**:
       - **From MSE**:  
         $\displaystyle \nabla_{\theta_j} \text{MSE} = -\text{error}^{(i)} \cdot X_j^{(i)} \quad (j \geq 1)$  
         $\displaystyle \nabla_{\theta_0} \text{MSE} = -\text{error}^{(i)} \quad \text{(bias term)}$
       - **From L2 Regularization** (for $j \geq 1$):  
         $\displaystyle \nabla_{\theta_j} \text{L2} = 2\lambda_2 \theta_j$
       - **From L1 Regularization** (for $j \geq 1$):  
         $\displaystyle \nabla_{\theta_j} \text{L1} = \lambda_1 \cdot \text{sign}(\theta_j)$  
         *(Note: The bias term is typically not regularized.)*
     - **Total gradient**:  
       $\displaystyle \nabla_{\theta_j} = \nabla_{\theta_j} \text{MSE} + \nabla_{\theta_j} \text{L1} + \nabla_{\theta_j} \text{L2}$
     - **Update weights** (using learning rate $\eta$):  
       $\displaystyle \theta_j := \theta_j - \eta \cdot \nabla_{\theta_j}$

3. **Early Stopping**:
   - Monitor the loss after each epoch.
   - If the improvement in loss between epochs is less than a predefined threshold (tolerance), stop training early.

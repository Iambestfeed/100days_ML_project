# Problem
Implement logistic regression using NumPy with mini-batch Stochastic Gradient Descent (SGD). The model should initialize weights, compute predictions using the sigmoid function, calculate Binary Cross-Entropy (BCE) loss, and update weights using gradient descent. The goal is to understand probabilistic classification and gradient descent mechanics.

# Knowledge

## 1. **Logistic Regression**
Logistic regression is used for binary classification problems where the output is either 0 or 1. Instead of directly predicting values like in linear regression, logistic regression predicts probabilities using the sigmoid function.

### **Formula:**  
\[ P(y=1 | X) = \sigma(Xw + b) \]  

Where:
- \( X \) is the feature matrix.
- \( w \) is the weight vector.
- \( b \) is the bias term.
- \( \sigma(z) \) is the sigmoid activation function:

\[ \sigma(z) = \frac{1}{1+e^{-z}} \]

## 2. **Binary Cross-Entropy (BCE) Loss**
The loss function for logistic regression is the Binary Cross-Entropy (BCE), which measures how well the predicted probabilities match the actual labels.

### **Formula:**  
\[ \text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \]  

Where:
- \( y_i \) is the true label (0 or 1).
- \( \hat{y}_i \) is the predicted probability \( \sigma(X_i w + b) \).
- \( n \) is the number of samples.

## 3. **Mini-Batch Stochastic Gradient Descent (SGD)**
### **Steps:**
1. **Initialization:** Randomly initialize weights \( w \) and bias \( b \).
2. **Iterate for multiple epochs:**
   - Shuffle data to prevent order bias.
   - **For each mini-batch:**
     - Compute predictions using the sigmoid function:
       \[ \hat{y} = \sigma(Xw + b) \]
     - Compute error:
       \[ \text{error} = \hat{y} - y \]
     - Compute gradient of BCE loss:
       \[ \nabla w = \frac{1}{m} X^T (\hat{y} - y) \]
       \[ \nabla b = \frac{1}{m} \sum (\hat{y} - y) \]
     - Update weights using learning rate \( \eta \):
       \[ w := w - \eta \cdot \nabla w \]
       \[ b := b - \eta \cdot \nabla b \]
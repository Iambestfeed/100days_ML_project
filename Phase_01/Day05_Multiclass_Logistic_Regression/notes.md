# Problem
Extend logistic regression to handle multiclass classification using Softmax Regression with mini-batch Stochastic Gradient Descent (SGD). The model should compute class probabilities using the softmax function, calculate Categorical Cross-Entropy loss, and update weights using gradient descent. The goal is to classify samples into multiple classes, demonstrated using the Iris dataset.

# Knowledge

## 1. **Softmax Regression**
Softmax regression (multinomial logistic regression) generalizes logistic regression to handle multiple classes. It predicts probability distributions over K classes.

### **Formula:**
\[ P(y=k|X) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \]

Where:
- \( X \) is the feature matrix
- \( z_k = X w_k + b_k \) is the linear combination for class k
- \( w_k \) is the weight vector for class k
- \( b_k \) is the bias term for class k
- K is the number of classes

## 2. **Categorical Cross-Entropy Loss**
The loss function measures the difference between predicted probability distributions and true class labels.

### **Formula:**
\[ \text{CCE} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik}) \]

Where:
- \( y_{ik} \) is the true label (one-hot encoded)
- \( \hat{y}_{ik} \) is the predicted probability for class k
- \( n \) is the number of samples
- \( K \) is the number of classes

## 3. **Mini-Batch SGD for Softmax Regression**
### **Steps:**
1. **Initialization:**
   - Initialize weights matrix \( W \) of shape (n_features, n_classes)
   - Convert labels to one-hot encoding

2. **Forward Pass:**
   - Compute linear combinations:
     \[ Z = XW \]
   - Apply softmax function:
     \[ \hat{Y} = \text{softmax}(Z) \]

3. **Backward Pass:**
   - Compute error:
     \[ \text{error} = \hat{Y} - Y \]
   - Compute gradients:
     \[ \nabla W = \frac{1}{m} X^T(\hat{Y} - Y) \]
   - Update weights:
     \[ W := W - \eta \cdot \nabla W \]
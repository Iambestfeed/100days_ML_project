## Problem

Our goal is to implement a **Linear Support Vector Machine (SVM)** that minimizes **hinge loss** using **Stochastic Gradient Descent (SGD)** and incorporates **L2 regularization** to prevent overfitting. The SVM aims to maximize the margin between two classes for binary classification. The hinge loss is defined as:

\[
\text{Hinge Loss: } \max(0, 1 - y(w^T x + b))
\]

where:
- \( y \in \{-1, 1\} \) are the binary class labels,
- \( w \) is the weight vector,
- \( x \) is the input feature vector,
- \( b \) is the bias term.

The objective is to optimize \( w \) and \( b \) such that the decision boundary maximizes the margin between the two classes while minimizing both misclassification error and model complexity (via L2 regularization).

---

## Knowledge

### 1. **Support Vector Machine (SVM)**
SVM is a supervised learning algorithm for binary classification. It works by finding the hyperplane that best separates the two classes with the largest margin. The margin is the distance between the hyperplane and the closest points from either class, called the **support vectors**.

#### **Key Characteristics:**
- SVMs aim to maximize the margin between classes.
- The hinge loss is used to penalize misclassifications and near-margin violations.
- L2 regularization ensures smaller weights, which helps prevent overfitting.

---

### 2. **Mathematical Formulation**

#### **Objective Function:**
The SVM optimization problem can be expressed as:

\[
\min_{w, b} \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))
\]

where:
- \( \|w\|^2 \) is the L2 regularization term,
- \( \lambda \) is the regularization strength,
- \( \max(0, 1 - y_i(w^T x_i + b)) \) is the hinge loss for each sample.

#### **Gradient Updates:**
The gradients for the weights (\( w \)) and bias (\( b \)) are derived based on the hinge loss and regularization:

1. For **weights** (\( w \)):
   - If \( 1 - y_i(w^T x_i + b) > 0 \):
     \[
     \nabla w = \lambda w - y_i x_i
     \]
   - Otherwise:
     \[
     \nabla w = \lambda w
     \]

2. For **bias** (\( b \)):
   - If \( 1 - y_i(w^T x_i + b) > 0 \):
     \[
     \nabla b = -y_i
     \]
   - Otherwise:
     \[
     \nabla b = 0
     \]

#### **Prediction Rule:**
The SVM predicts class labels using the sign of the decision function:
\[
y^* = \text{sign}(w^T x + b)
\]

---

### 3. **Hinge Loss**
The hinge loss penalizes misclassified points and points that lie within the margin. It is defined as:
\[
\text{hinge}(y, f(x)) = \max(0, 1 - y(w^T x + b))
\]

#### **Behavior:**
- If \( y(w^T x + b) \geq 1 \), the loss is 0 (correct classification with margin).
- If \( y(w^T x + b) < 1 \), the loss increases linearly with the distance from the margin.

---

### 4. **Stochastic Gradient Descent (SGD)**
SGD is an optimization algorithm that updates model parameters incrementally using randomly selected samples (or mini-batches) instead of the entire dataset.

#### **Advantages of SGD:**
- Faster convergence for large datasets.
- Efficient memory usage for online learning.

#### **Weight Updates:**
At each iteration, the weight vector \( w \) and bias \( b \) are updated using the computed gradients:
\[
w = w - \eta \cdot \nabla w
\]
\[
b = b - \eta \cdot \nabla b
\]
where:
- \( \eta \) is the learning rate.

---
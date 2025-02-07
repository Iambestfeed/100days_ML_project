## Problem
Implement a Perceptron model that learns to classify linearly separable data using the perceptron learning rule. The model should update weights based on misclassified examples and use a step activation function for predictions.

## Knowledge

### 1. **Perceptron Model**
A perceptron is a type of linear classifier that maps input features to binary labels using a weighted sum and a step activation function. It is a fundamental model in machine learning for learning linearly separable functions.

### 2. **Mathematical Formulation**
#### **Prediction Rule:**
The perceptron makes predictions using the step function:
\[ y^* = 1 \text{ if } w^T x + b \geq 0, \text{ otherwise } 0 \]
where:
- \( w \) is the weight vector,
- \( x \) is the input feature vector,
- \( b \) is the bias term.

#### **Weight Update Rule:**
Weights are updated using the perceptron learning rule:
\[ w = w + \eta (y - y^*) x \]
\[ b = b + \eta (y - y^*) \]
where:
- \( \eta \) is the learning rate,
- \( y \) is the true label,
- \( y^* \) is the predicted label.

### 3. **Step Activation Function**
The step function determines the perceptron’s output:
\[ f(z) = \begin{cases} 1, & z \geq 0 \\ 0, & z < 0 \end{cases} \]
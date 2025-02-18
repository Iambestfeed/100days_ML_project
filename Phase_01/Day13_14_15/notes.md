## Problem

The goal of this implementation is to create a **2-layer neural network** for **classification** tasks. The model minimizes the **Mean Squared Error (MSE)** loss function with **L2 regularization** to prevent overfitting. The network is designed with one hidden layer, and it utilizes **ReLU** (Rectified Linear Unit) activation for the hidden layer, while the output layer uses a linear activation function.

---

## Knowledge

### 1. **Multi-Layer Perceptron (MLP)**

An MLP is a type of neural network that consists of an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to all neurons in the adjacent layers. The main goal is to learn a mapping from input features to the target output.

#### **Key Characteristics:**
- **Input Layer**: Receives the feature vectors.
- **Hidden Layer**: Contains neurons with ReLU activation to introduce non-linearity.
- **Output Layer**: Produces the final prediction. The output layer can be linear for regression or softmax for classification tasks.
- **Activation Function**: The **ReLU** function is used in the hidden layer to introduce non-linearity, which enables the network to learn complex patterns.
- **L2 Regularization**: Prevents overfitting by adding a penalty for large weights, encouraging smaller weight values.
- **Mini-batch Gradient Descent**: Uses small, random subsets of the dataset for gradient updates to improve efficiency and convergence.

---

### 2. **Mathematical Formulation**

#### **Objective Function (Loss Function):**
The **Mean Squared Error (MSE)** is used to measure the difference between the predicted output and the actual target values:

\[
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
\]

where:
- \( y_i \) is the true target label,
- \( \hat{y_i} \) is the predicted output of the network.

L2 regularization is added to the loss function to penalize large weights:

\[
\text{Loss}_{\text{total}} = \text{Loss} + \frac{\lambda}{2} \sum_{j} w_j^2
\]

where:
- \( \lambda \) is the regularization strength,
- \( w_j \) represents the weights.

#### **Forward Propagation:**
1. **Hidden Layer**: The input is multiplied by the weights and added to the bias:
   \[
   \text{hidden\_input} = X \cdot W_1^T + b_1
   \]
   The **ReLU** activation is then applied:
   \[
   \text{hidden\_output} = \text{ReLU}(\text{hidden\_input})
   \]

2. **Output Layer**: The hidden layer output is used to compute the final predictions:
   \[
   \text{output\_input} = \text{hidden\_output} \cdot W_2^T + b_2
   \]

#### **Backpropagation:**
The gradients for the weights and biases are computed based on the loss function and are used to update the parameters.

1. **Output Layer Gradients**:
   \[
   d_{\text{output}} = 2 \cdot (\hat{y} - y) / m
   \]
   Gradients for \( W_2 \) and \( b_2 \) are calculated:
   \[
   dW_2 = \text{hidden\_output}^T \cdot d_{\text{output}} + \lambda W_2
   \]
   \[
   db_2 = \sum d_{\text{output}}
   \]

2. **Hidden Layer Gradients**:
   Using the derivative of the **ReLU** function:
   \[
   d_{\text{hidden}} = d_{\text{output}} \cdot W_2 \cdot (\text{hidden\_output} > 0)
   \]
   Gradients for \( W_1 \) and \( b_1 \) are computed:
   \[
   dW_1 = X^T \cdot d_{\text{hidden}} + \lambda W_1
   \]
   \[
   db_1 = \sum d_{\text{hidden}}
   \]

#### **Gradient Updates:**
The parameters are updated using the gradients computed during backpropagation:
\[
W_1 = W_1 - \eta \cdot dW_1
\]
\[
b_1 = b_1 - \eta \cdot db_1
\]
\[
W_2 = W_2 - \eta \cdot dW_2
\]
\[
b_2 = b_2 - \eta \cdot db_2
\]
where:
- \( \eta \) is the learning rate.

---

### 3. **ReLU Activation Function**
The **ReLU** (Rectified Linear Unit) activation function is defined as:
\[
\text{ReLU}(x) = \max(0, x)
\]
ReLU introduces non-linearity into the model, enabling it to learn complex patterns. It is widely used in neural networks because of its simplicity and effectiveness.

#### **Behavior:**
- For positive input, the output is the same.
- For negative input, the output is 0.


### **4. Dropout Regularization**  

#### **Definition:**  
Dropout is a regularization technique that randomly sets a fraction of the neurons to zero during training to prevent overfitting. By doing this, the network does not overly rely on specific neurons and instead learns more robust and generalizable patterns.  

#### **Mathematical Formulation:**  
Given an activation \( a \) in a hidden layer, dropout randomly masks it as follows:  

\[
a_{\text{drop}} = \frac{a \cdot M}{1 - p}
\]

where:  
- \( p \) is the dropout rate (probability of dropping a neuron).  
- \( M \) is a binary mask drawn from a Bernoulli distribution \( M \sim \text{Bernoulli}(1 - p) \).  
- The division by \( 1 - p \) ensures that the expected sum of activations remains the same during training and inference.  

### **5. Batch Normalization**  

#### **Definition:**  
Batch Normalization (BatchNorm) is a technique used to normalize the inputs of each layer to stabilize and accelerate training. It helps address the problem of internal covariate shift, where the distribution of inputs to each layer changes over time.  

#### **Mathematical Formulation:**  
Given a mini-batch \( X \), BatchNorm transforms the inputs as follows:  

\[
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
\]

\[
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
\]

\[
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\]

\[
y_i = \gamma \hat{x}_i + \beta
\]

where:  
- \( \mu_B \) is the batch mean.  
- \( \sigma_B^2 \) is the batch variance.  
- \( \hat{x}_i \) is the normalized input.  
- \( \gamma \) and \( \beta \) are learnable scale and shift parameters.  
- \( \epsilon \) is a small constant for numerical stability.  
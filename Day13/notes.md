## Dropout and Batch Normalization in Neural Networks  

### **1. Dropout Regularization**  

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

### **2. Batch Normalization**  

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
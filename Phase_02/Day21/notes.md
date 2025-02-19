# Problem
Implement a Recurrent Neural Network (RNN) using NumPy to process sequential data and predict the next token in a sequence. The goal is to create a flexible RNN class that supports configurable activation functions and embedding types, enabling it to handle tasks like next-token prediction for text data.

# Knowledge

## 1. Recurrent Neural Networks (RNNs)
RNNs are designed to model sequential data by maintaining a hidden state that captures information from previous timesteps. This makes them suitable for tasks involving sequences, such as language modeling, time series prediction, and text generation.

- **Core Components**:
  - **Input**: A sequence of tokens or features, represented numerically (e.g., as indices or embeddings).
  - **Hidden State (\( h_t \))**: A vector that summarizes the information from all previous inputs up to time \( t \). It is updated at each timestep based on the current input and the previous hidden state.
  - **Output**: The prediction for the current timestep, such as the probability distribution over a vocabulary for next-token prediction.
  - **Weights and Biases**: Parameters that transform inputs and hidden states into new hidden states and outputs.

- **Update Equation**:
  At each timestep \( t \):
  - Hidden state update: \( h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b) \)
  - Output: \( o_t = W_{hy}h_t + b_y \)

  Where:
  - \( x_t \): Input at time \( t \) (e.g., embedded token).
  - \( h_{t-1} \): Previous hidden state.
  - \( W_{xh} \): Weight matrix from input to hidden layer.
  - \( W_{hh} \): Weight matrix from hidden to hidden layer.
  - \( W_{hy} \): Weight matrix from hidden to output layer.
  - \( b \): Bias for hidden state.
  - \( b_y \): Bias for output.
  - \( f \): Activation function (e.g., tanh or ReLU).

This structure allows the RNN to maintain a memory of past inputs, making it effective for sequential tasks but potentially vulnerable to issues like vanishing gradients for long sequences.

## 2. Activation Functions
Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.

- **Tanh**: Maps inputs to the range (-1, 1), centering the data and helping with gradient flow. It is defined as \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \).
- **ReLU**: Sets all negative values to 0 and keeps positive values unchanged (\( \text{ReLU}(x) = \max(0, x) \)). It is computationally efficient but can lead to "dying neurons" if gradients become too small.

## 3. Embedding Types
Embeddings transform discrete tokens (e.g., words) into continuous vectors that the network can process.

- **One-Hot Encoding**: Represents each token as a vector with a 1 at the index of the token and 0s elsewhere. It is simple but high-dimensional and sparse.
- **Random Initialization**: Assigns random vectors to each token, which can be fine-tuned during training. This is denser and more flexible but requires more computation.

## 4. Next Token Prediction
The task here is to predict the most likely next token in a sequence based on previous tokens. This involves:
- Tokenizing the input text into numerical indices.
- Passing the sequence through the RNN to generate hidden states.
- Using the final hidden state to compute output probabilities over a vocabulary, selecting the most likely token via argmax.
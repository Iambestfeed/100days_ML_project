# Problem
Implement a Gated Recurrent Unit (GRU) network using NumPy to process sequential data and predict the next token in a sequence. The goal is to extend a basic Recurrent Neural Network (RNN) by incorporating update and reset gates to manage dependencies in sequential tasks like next-token prediction, offering a simpler alternative to LSTMs while still addressing the vanishing gradient problem and improving performance on sequential data.

# Knowledge
## Gated Recurrent Unit (GRU) Networks
GRUs are a type of Recurrent Neural Network designed to effectively model dependencies in sequential data, balancing simplicity and performance compared to traditional RNNs and LSTMs. Unlike LSTMs, GRUs use two gates—update and reset—to control information flow, merging the cell state and hidden state into a single hidden state for reduced computational complexity.

- **Core Components**:
  - **Hidden State (\( h_t \))**: Represents both the memory and output of the GRU at each timestep, passed to the next timestep and used for predictions.
  - **Gates**:
    - **Update Gate (\( z_t \))**: Determines how much of the previous hidden state to retain and how much new information to incorporate.
    - **Reset Gate (\( r_t \))**: Controls how much of the previous hidden state influences the candidate hidden state, allowing the model to "forget" irrelevant past information.
  - **Candidate Hidden State (\( \tilde{h}_t \))**: A proposed update to the hidden state based on the current input and a gated version of the previous hidden state.

- **Update Equations**:
  At each timestep \( t \):
  - Update gate: \( z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z) \)
  - Reset gate: \( r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r) \)
  - Candidate hidden state: \( \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \)
  - Hidden state update: \( h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \)

  Where:
  - \( x_t \): Input at time \( t \).
  - \( h_{t-1} \): Previous hidden state.
  - \( W_z, W_r, W_h \): Weight matrices from input to update gate, reset gate, and candidate hidden state, respectively.
  - \( U_z, U_r, U_h \): Weight matrices from hidden state to update gate, reset gate, and candidate hidden state, respectively.
  - \( b_z, b_r, b_h \): Biases for update gate, reset gate, and candidate hidden state.
  - \( \sigma \): Sigmoid activation function (outputs values between 0 and 1).
  - \( \tanh \): Hyperbolic tangent activation function (outputs values between -1 and 1).
  - \( \odot \): Element-wise multiplication.

These equations enable the GRU to selectively update its hidden state, balancing the retention of past information and the integration of new inputs, making it efficient and effective for sequential tasks.
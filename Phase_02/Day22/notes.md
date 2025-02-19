# Problem
Implement a Long Short-Term Memory (LSTM) network using NumPy to process sequential data, manage long-term dependencies, and predict the next token in a sequence. The goal is to extend a basic Recurrent Neural Network (RNN) to include input, forget, and output gates, as well as a cell state, to address the vanishing gradient problem and improve performance on sequential tasks like next-token prediction.

# Knowledge

## 1. Long Short-Term Memory (LSTM) Networks
LSTMs are a type of Recurrent Neural Network designed to model long-term dependencies in sequential data more effectively than traditional RNNs. They use a memory cell and three key gates to control the flow of information: the input gate, forget gate, and output gate.

- **Core Components**:
  - **Cell State (\( c_t \))**: Acts as the memory of the network, allowing information to persist over long sequences.
  - **Hidden State (\( h_t \))**: The output of the LSTM at each timestep, used for predictions and passed to the next timestep.
  - **Gates**:
    - **Input Gate (\( i_t \))**: Decides what new information to add to the cell state.
    - **Forget Gate (\( f_t \))**: Determines what information to discard from the previous cell state.
    - **Output Gate (\( o_t \))**: Controls what information from the cell state is used to compute the hidden state.

- **Update Equations**:
  At each timestep \( t \):
  - Input gate: \( i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i) \)
  - Forget gate: \( f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f) \)
  - Output gate: \( o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o) \)
  - Candidate cell: \( g_t = \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c) \)
  - Cell state update: \( c_t = f_t \odot c_{t-1} + i_t \odot g_t \)
  - Hidden state update: \( h_t = o_t \odot \tanh(c_t) \)

  Where:
  - \( x_t \): Input at time \( t \).
  - \( h_{t-1} \): Previous hidden state.
  - \( c_{t-1} \): Previous cell state.
  - \( W \) and \( b \): Weight matrices and biases for each gate and candidate cell.
  - \( \sigma \): Sigmoid activation function (outputs values between 0 and 1).
  - \( \tanh \): Hyperbolic tangent activation function (outputs values between -1 and 1).
  - \( \odot \): Element-wise multiplication.

These equations allow the LSTM to selectively remember or forget information, mitigating the vanishing gradient problem by ensuring gradients can flow through the cell state without diminishing.

## 2. Vanishing Gradient Problem
In traditional RNNs, gradients can become extremely small (vanish) during backpropagation through time, making it difficult to learn long-term dependencies. LSTMs address this by:
- Using the cell state as a conveyor belt for information, which can remain constant or change gradually.
- Employing gates to control what information is retained or discarded, ensuring that important information persists over many timesteps.
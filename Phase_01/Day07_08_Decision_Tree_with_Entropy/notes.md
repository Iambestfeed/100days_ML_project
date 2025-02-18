# Problem
Implement a decision tree using NumPy that recursively splits data based on entropy. The model should support tree depth control and leaf node constraints. The goal is to interpret splits via feature importance and compare entropy-based splits with Gini-based splits.

# Knowledge

## 1. **Decision Trees for Classification**
A decision tree is a flowchart-like structure where each internal node represents a decision based on a feature, each branch represents an outcome, and each leaf node represents a class label.

## 2. **Entropy**
Entropy measures the impurity or disorder in a dataset. It is used to determine the best feature for splitting the data.

### **Formula:**  
\[ H = - \sum p_i \log p_i \]  

Where:
- \( p_i \) is the proportion of samples belonging to class \( i \) in the subset.
- \( H \) ranges from 0 (pure split) to a maximum value when classes are equally distributed.
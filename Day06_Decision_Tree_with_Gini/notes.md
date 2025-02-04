# Problem
Implement a decision tree using NumPy that recursively splits data based on Gini impurity. The model should support tree depth control and leaf node constraints. The goal is to interpret splits via feature importance.

# Knowledge

## 1. **Decision Trees for Classification**
A decision tree is a flowchart-like structure where each internal node represents a decision based on a feature, each branch represents an outcome, and each leaf node represents a class label.

## 2. **Gini Impurity**
Gini impurity measures the likelihood of incorrect classification if a sample was randomly classified according to the distribution of class labels in a subset.

### **Formula:**  
\[ G = 1 - \sum p_i^2 \]  

Where:
- \( p_i \) is the proportion of samples belonging to class \( i \) in the subset.
- \( G \) ranges from 0 (pure split) to 0.5 (worst case with two balanced classes).

## 3. **Recursive Splitting**
### **Steps:**
1. **Start with the full dataset** as the root node.
2. **For each feature, compute Gini impurity** for possible split points.
3. **Choose the split** that minimizes weighted Gini impurity in child nodes.
4. **Recursively repeat** for each child node until stopping criteria are met.

## 4. **Stopping Criteria**
- **Maximum tree depth**: Limits tree growth.
- **Minimum samples per leaf**: Ensures each leaf node has sufficient data.
- **Pure node condition**: Stops splitting if all samples in a node belong to the same class.


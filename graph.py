import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    'Decision Tree', 'DNN', 'LDA', 'Logistic Regression',
    'Naive Bayes', 'Random Forest', 'RNN', 'SVM'
]

# Corresponding accuracy and F1 scores
accuracy_scores = [1.00, 0.35, 0.0201, 0.5031, 0.0352, 1.00, 0.34, 0.2608]
f1_scores = [1.00, 0.26, 0.0104, 0.0236, 0.0181, 1.00, 0.344, 0.1116]

# Plotting
x = np.arange(len(models))
bar_width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width/2, accuracy_scores, bar_width, label='Accuracy', color='skyblue')
plt.bar(x + bar_width/2, f1_scores, bar_width, label='F1 Score', color='orange')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Comparison: Accuracy vs F1 Score')
plt.xticks(x, models, rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

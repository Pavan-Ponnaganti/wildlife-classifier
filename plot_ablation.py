import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data points illustrating ablation of frozen layers on EfficientNet-B0
# Quantifying "unfrozen layers" using Trainable Parameters (in Millions)
ablation_stages = [
    'Classifier Only\n(0 Blocks)', 
    'Top Layers Unfrozen\n(1 Block)', 
    'Mid-Top Unfrozen\n(2 Blocks)', 
    'Half Network Unfrozen\n(4 Blocks)', 
    'Full Fine-Tune\n(All Blocks)'
]

trainable_params_millions = [0.24, 1.5, 2.5, 3.2, 4.1]

# Typical accuracy curve: increases as we unfreeze top semantic layers, 
# then plateaus or drops slightly due to overfitting the small generic features layers.
test_accuracy = [82.2, 84.6, 86.1, 86.3, 85.1]  

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create a line plot with markers
ax = sns.lineplot(
    x=trainable_params_millions, 
    y=test_accuracy, 
    marker='o', 
    markersize=10, 
    linewidth=2.5, 
    color='#1f77b4'
)

# Annotate each point with the stage description and exact accuracy
for i, stage in enumerate(ablation_stages):
    # Offset alternating points up or down to avoid label overlap
    y_offset = 15 if i % 2 == 0 else -30
    
    ax.annotate(
        f"{stage}\n{test_accuracy[i]}%", 
        (trainable_params_millions[i], test_accuracy[i]),
        textcoords="offset points",
        xytext=(0, y_offset), 
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9)
    )

plt.title('Ablation Study: Trainable Parameters (Unfrozen) vs. Model Accuracy', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Number of Trainable Parameters (Millions)', fontsize=12, fontweight='bold')
plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')

# Give Y-axis padding for text
plt.ylim(80, 90)
plt.xlim(0, 4.5)

plt.tight_layout()
plt.savefig('ablation_study_graph.png', dpi=300)
print("Graph successfully saved as 'ablation_study_graph.png'")

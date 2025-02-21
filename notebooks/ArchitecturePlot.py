import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define model architecture components
components = [
    "Encoder\n(ResNet, EfficientNet, etc.)",
    "Global average pooling (GAP)",
    "Flatten",
    "Fully connected\n(4096 neurons)",
    "ReLU activation",
    "Dropout (0.6)",
    "Fully connected\n(\"2 * number of keypoints\" neurons)"
]



# Create figure
fig, ax = plt.subplots(figsize=(6, 8))

# Define box parameters
box_width = 3
box_height = 0.6
y_positions = list(range(len(components), 0, -1))

# Adjust arrow length by increasing spacing between components
spacing_factor = 1  # Increase spacing between components
y_positions = [y * spacing_factor for y in y_positions]

# Plot the architecture
for i, (comp, y) in enumerate(zip(components, y_positions)):
    # Set specific colors for certain layers
    if "Encoder" in comp:
        facecolor = "lightgreen"  # Light green for Encoder
    elif "Fully connected" in comp:
        facecolor = "lightblue"  # Light blue for FC layers
    else:
        facecolor = "lightgray"  # Default color for other layers

    ax.add_patch(mpatches.FancyBboxPatch(
        (-box_width / 2, y - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.1",
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.5
    ))
    ax.text(0, y, comp, ha="center", va="center", fontsize=10, fontweight="bold")

# Draw arrows between layers
for i in range(len(components) - 1):
    ax.annotate("", xy=(0, y_positions[i] - box_height / 2), xytext=(0, y_positions[i + 1] + box_height / 2),
                arrowprops=dict(arrowstyle="<-", lw=1.5, color="black"))

# Remove axes
ax.set_xlim(-2, 2)
ax.set_ylim(0, max(y_positions) + 1)
ax.axis("off")

# Show the figure
plt.title("Regression-based model architecture (based on DeepPose)", fontsize=12, fontweight="bold")
plt.show()

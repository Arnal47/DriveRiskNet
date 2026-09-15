def save_learning_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="weighted loss", title="Loss"); axes[0].legend()
    axes[1].plot(epochs, [r["train_f1_macro"] for r in history], label="train")
    axes[1].plot(epochs, [r["val_f1_macro"] for r in history], label="validation")
    axes[1].set(xlabel="epoch", ylabel="macro-F1", title="Macro-F1", ylim=(0, 1)); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_confusion_matrix(matrix, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .config import CLASS_NAMES
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    ax.set_xticks(range(3), CLASS_NAMES); ax.set_yticks(range(3), CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title=title); fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

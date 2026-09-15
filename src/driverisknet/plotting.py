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

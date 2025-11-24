def visualize_models(X, feature_names, models, model_names, chromosomes, output_path):
    """
    Visualize feature effects for interpretable models.
    - X: input features
    - feature_names: list of feature names
    - models: list of models [GA-GAM, Decision Tree, baseline PyGAM]
    - model_names: list of model names
    - chromosomes: list of chromosomes corresponding to GA-GAM
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features, len(models), figsize=(5*len(models), 4*n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    for f_idx in range(n_features):
        x_vals = np.linspace(X[:, f_idx].min(), X[:, f_idx].max(), 100)
        X_plot = np.tile(np.mean(X, axis=0), (100,1))

        for m_idx, model in enumerate(models):
            ax = axes[f_idx, m_idx]

            # Handle GA-GAM with chromosome
            if hasattr(model, "partial_dependence"):
                # For GA-GAM, find if this feature is active
                chromosome = chromosomes[m_idx]  # GA-GAM chromosome only
                if chromosome.genes[f_idx]["type"] == "none":
                    ax.text(0.5, 0.5, "Inactive", ha="center", va="center", fontsize=12)
                else:
                    # Map feature index to GAM term index
                    term_idx = 0
                    for g in chromosome.genes[:f_idx]:
                        if g["type"] != "none":
                            term_idx += 1
                    pd_result, confi = model.partial_dependence(term=term_idx, width=0.95)
                    pd_result = np.asarray(pd_result)
                    if pd_result.ndim == 1:
                        ax.plot(x_vals, pd_result, label="Effect")
                        if confi is not None:
                            ax.fill_between(x_vals, confi[:,0], confi[:,1], alpha=0.2)
                    else:
                        ax.plot(pd_result[:,0], pd_result[:,1], label="Effect")
                        ax.fill_between(pd_result[:,0], confi[:,0], confi[:,1], alpha=0.2)
            
            else:  # Decision Tree
                X_plot[:, f_idx] = x_vals
                y_pred = model.model.predict(X_plot)
                ax.plot(x_vals, y_pred, label="Approx. effect")

            ax.set_title(f"{model_names[m_idx]}: {feature_names[f_idx]}")
            ax.set_xlabel(feature_names[f_idx])
            ax.set_ylabel("Contribution / Prediction")
            ax.grid(True)

    plt.tight_layout()
    plt.show()
    fig.savefig(output_path)

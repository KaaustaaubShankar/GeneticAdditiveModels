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

    import os

    n_features = X.shape[1]

    # For each feature (row), create a separate horizontal plot containing one subplot per model
    for f_idx in range(n_features):
        fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
        # Ensure axes is iterable
        if len(models) == 1:
            axes = np.array([axes])

        x_vals = np.linspace(X[:, f_idx].min(), X[:, f_idx].max(), 100)
        X_plot = np.tile(np.mean(X, axis=0), (100, 1))

        for m_idx, model in enumerate(models):
            ax = axes[m_idx]

            # Handle GA-GAM with chromosome
            annotation = None

            if hasattr(model, "partial_dependence"):
                # For GA-GAM, find if this feature is active
                chromosome = chromosomes[m_idx]
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
                            ax.fill_between(x_vals, confi[:, 0], confi[:, 1], alpha=0.2)
                    else:
                        ax.plot(pd_result[:, 0], pd_result[:, 1], label="Effect")
                        if confi is not None:
                            ax.fill_between(pd_result[:, 0], confi[:, 0], confi[:, 1], alpha=0.2)

                    # Annotation from chromosome (knots/lambda)
                    gene = chromosome.genes[f_idx]
                    if gene.get("type") == "spline":
                        k = gene.get("knots")
                        lam = gene.get("lambda")
                        annotation = f"knots: {k}\nλ={lam:.3g}"
                    elif gene.get("type") == "linear":
                        # Get actual lambda from the fitted GAM term
                        try:
                            term = model.terms[term_idx]
                            lam_val = term.lam
                            if isinstance(lam_val, (list, np.ndarray)):
                                lam_val = float(np.mean(lam_val))
                            annotation = f"linear\nλ={lam_val:.2g}"
                        except Exception:
                            annotation = "linear"

                    else:
                        annotation = "none"

            else:  # Decision Tree or baseline container
                X_plot[:, f_idx] = x_vals
                # If this is a baseline container with an inner `model` (e.g., PyGAMRegressionBaseline)
                if hasattr(model, "model") and model.model is not None and hasattr(model.model, "predict"):
                    y_pred = model.model.predict(X_plot)
                    ax.plot(x_vals, y_pred, label="Approx. effect")

                    # Try to annotate using the inner LinearGAM terms
                    try:
                        term = model.model.terms[f_idx]
                        n_spl = getattr(term, "n_splines", None)
                        lam_val = term.lam if hasattr(term, "lam") else None
                        if isinstance(lam_val, (list, np.ndarray)):
                            lam_val = np.mean(lam_val)
                        if n_spl is not None and lam_val is not None:
                            annotation = f"n_splines: {int(n_spl)}\nλ={float(lam_val):.3g}"
                    except Exception:
                        # Fallback: try to read attribute `n_splines` from model container
                        try:
                            if hasattr(model, "n_splines"):
                                annotation = f"n_splines: {int(model.n_splines)}"
                        except Exception:
                            pass
                else:
                    # Pure decision tree container
                    y_pred = model.model.predict(X_plot)
                    ax.plot(x_vals, y_pred, label="Approx. effect")

            ax.set_title(f"{model_names[m_idx]}: {feature_names[f_idx]}")
            ax.set_xlabel(feature_names[f_idx])
            ax.set_ylabel("Contribution / Prediction")
            ax.grid(True)

            # Draw annotation if available
            if annotation:
                ax.text(
                    0.98,
                    0.02,
                    annotation,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
                )

        plt.tight_layout()

        # Save each feature plot separately. If output_path is a directory, save inside it.
        safe_name = feature_names[f_idx].replace(" ", "_")
        if os.path.isdir(output_path):
            save_path = os.path.join(output_path, f"{safe_name}.png")
        else:
            base, ext = os.path.splitext(output_path)
            if ext == "":
                ext = ".png"
            save_path = f"{base}_{safe_name}{ext}"

        fig.savefig(save_path)
        plt.show()
        plt.close(fig)
def visualize_pareto_front(pareto_front, output_path):
    """
    Visualize the Pareto front of solutions.
    - pareto_front: list of individuals in the Pareto front
    - output_path: path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    r = [ind.fitness.values[0] for ind in pareto_front]  # normalized RMSE
    p = [ind.fitness.values[1] for ind in pareto_front]  # penalty in [0,1]
    plt.scatter(p, r)
    plt.xlabel("Complexity penalty")
    plt.ylabel("Normalized RMSE")
    plt.title("Pareto front: complexity vs normalized RMSE")
    plt.savefig(output_path)
    plt.show()
    plt.close()

    
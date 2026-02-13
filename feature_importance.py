import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_linear_coefficients(
    model,
    feature_names,
    top_n=15
):
    """
    Extract and rank coefficients from linear models
    (Ridge, Lasso, LinearRegression).

    Parameters:
    - model: trained linear model
    - feature_names: list of feature column names
    - top_n: number of top features to return

    Returns:
    - DataFrame of coefficients sorted by importance
    """

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_,
        "Absolute Importance": np.abs(model.coef_)
    })

    coef_df = coef_df.sort_values(
        by="Absolute Importance",
        ascending=False
    )

    return coef_df.head(top_n)


def plot_tree_feature_importance(
    model,
    feature_names,
    top_n=15,
    model_name="Tree-Based Model"
):
    """
    Plot feature importances for tree-based models.

    Parameters:
    - model: trained tree-based model
    - feature_names: list or Index of feature names
    - top_n: number of top features to display
    - model_name: name used in the plot title
    """

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    ).head(top_n)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1]
    )
    plt.xlabel("Feature Importance")
    plt.title(f"{model_name}: Feature Importance")
    plt.tight_layout()
    plt.show()

    return importance_df


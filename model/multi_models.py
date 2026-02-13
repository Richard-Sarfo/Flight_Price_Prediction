import logging
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def train_and_compare_regressors(
    X_train,
    X_test,
    y_train,
    y_test,
    random_state=42,
    include_gb=True
):
    """
    Train and evaluate multiple regression models using GridSearchCV.

    Returns:
    - results DataFrame
    - trained models dictionary
    """

    models = {
        "Ridge Regression": {
            "model": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0]}
        },
        "Lasso Regression": {
            "model": Lasso(max_iter=5000),
            "params": {"alpha": [0.01, 0.1, 1.0]}
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=random_state),
            "params": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=random_state),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20]
            }
        }
    }

    if include_gb:
        models["Gradient Boosting"] = {
            "model": GradientBoostingRegressor(random_state=random_state),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1]
            }
        }

    results = []
    trained_models = {}

    for name, cfg in models.items():
        logger.info(f"Training {name} with GridSearchCV")

        #SAFETY: enforce dict structure
        if not isinstance(cfg, dict):
            cfg = {"model": cfg, "params": {}}

        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Best Params": grid.best_params_
        })

        trained_models[name] = best_model

        logger.info(
            f"{name} | R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="RMSE", ascending=True)
        .reset_index(drop=True)
    )

    return results_df, trained_models


def demonstrate_regularization_effect(
    X_train,
    X_test,
    y_train,
    y_test,
    model_type="ridge",
    alphas=None
):
    """
    Demonstrate how Ridge or Lasso regularization reduces overfitting
    using train vs test error.

    Returns:
    - DataFrame with alpha, train RMSE, test RMSE
    """

    if alphas is None:
        alphas = np.logspace(-4, 2, 10)

    results = []

    for alpha in alphas:
        if model_type == "ridge":
            model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            model = Lasso(alpha=alpha, max_iter=5000)
        else:
            raise ValueError("model_type must be 'ridge' or 'lasso'")

        model.fit(X_train, y_train)

        train_rmse = np.sqrt(
            mean_squared_error(y_train, model.predict(X_train))
        )
        test_rmse = np.sqrt(
            mean_squared_error(y_test, model.predict(X_test))
        )

        results.append({
            "Model": model_type.capitalize(),
            "Alpha": alpha,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Gap (Variance)": test_rmse - train_rmse
        })

        logger.info(
            f"{model_type.capitalize()} | alpha={alpha:.4f} | "
            f"Train RMSE={train_rmse:.2f}, Test RMSE={test_rmse:.2f}"
        )

    return pd.DataFrame(results)

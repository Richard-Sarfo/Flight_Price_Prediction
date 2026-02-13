import logging
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model
):
    """
    Train and evaluate a linear regression model.

    Returns:
    - trained model
    - metrics dictionary (R2, MAE, RMSE)
    """

    logger.info("Training Linear Regression model")

    # Train
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Predict
    y_pred = model.predict(X_test)
    logger.info("Predictions completed")

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")

    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }

    return model, metrics

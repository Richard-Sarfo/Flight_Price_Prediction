import matplotlib.pyplot as plt
import seaborn as sns

def plot_fare_distributions(df):
    cols = ['Total Fare (BDT)', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)']
    
    for col in cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

def boxplot_fare_by_airline(df):
    plt.figure(figsize=(10,5))
    sns.boxplot(x='Airline', y='Total Fare (BDT)', data=df)
    plt.xticks(rotation=45)
    plt.title('Fare Variation Across Airlines')
    plt.show()

def plot_avg_fare_over_time(df, time_col):
    avg_fare = df.groupby(time_col)['Total Fare (BDT)'].mean()
    
    plt.figure(figsize=(8,4))
    avg_fare.plot(marker='o')
    plt.ylabel('Average Fare')
    plt.title(f'Average Fare by {time_col}')
    plt.grid(True)
    plt.show()

def correlation_heatmap(df):
    corr = df.select_dtypes(include='number').corr()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')

    plt.show()

    # Model Interpretation Visualizations

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()]
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()

def plot_regularization_effect(
    df,
    model_name="Ridge Regression"
):
    """
    Plot Train vs Test RMSE across different alpha values
    to demonstrate overfitting reductio

    Parameters:
    - df: DataFrame returned from demonstrate_regularization_effect
    - model_name: Name to display in the plot title
    """

    plt.figure(figsize=(8, 5))

    plt.plot(df["Alpha"], df["Train RMSE"], label="Train RMSE")
    plt.plot(df["Alpha"], df["Test RMSE"], label="Test RMSE")

    plt.xscale("log")
    plt.xlabel("Alpha (Regularization Strength)")
    plt.ylabel("RMSE")
    plt.title(f"{model_name}: Bias–Variance Tradeoff")

    plt.legend()
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt

def plot_results(y_test, y_pred):
    """
    Plots the actual vs predicted values and residuals for the model.
    """
    # Plot actual vs predicted closing prices
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')  # Actual values
    plt.scatter(range(len(y_pred)), y_pred, color='orange', label='Predicted')  # Predicted values
    plt.title("Actual vs Predicted Closing Prices")
    plt.xlabel("Test Data Index")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.show()

    # Plot residuals (errors between actual and predicted values)
    residuals = y_test - y_pred  # Difference between actual and predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, color='red')  # Residual values
    plt.axhline(0, color='black', linestyle='--')  # Reference line at zero
    plt.title("Residuals vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.show()

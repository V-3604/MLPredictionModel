from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model on the provided training data.
    """
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Save the trained model for future use
    joblib.dump(model, 'google_stock_model.pkl')  # Save to a file
    print("Model trained and saved as google_stock_model.pkl")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and calculates performance metrics.
    """
    # Predict closing prices for the test dataset
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics: MAE, RMSE, and R² score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Store metrics in a dictionary for easy reporting
    metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Root Mean Squared Error (RMSE)': rmse,
        'R² Score': r2
    }
    return metrics, y_pred

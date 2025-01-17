from preprocessing import fetch_data, preprocess_data
from modeling import train_model, evaluate_model
from visualization import plot_results

# Fetch historical stock data for Google (GOOGL)
symbol = 'AAPL'
print("Fetching data...")
data = fetch_data(symbol)
print("Data Fetched. ")
# Process the data and prepare it for training and testing
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train the linear regression model
model = train_model(X_train, y_train)

# Evaluate the model with test data
metrics, y_pred = evaluate_model(model, X_test, y_test)

# Display evaluation metrics
print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Visualize actual vs predicted values and residuals
plot_results(y_test, y_pred)

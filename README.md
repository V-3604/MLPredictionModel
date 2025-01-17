# Stock Market Prediction Using Linear Regression

This repository contains a machine learning project focused on stock market prediction. It demonstrates the application of linear regression to forecast stock prices using historical data, while emphasizing clean data preprocessing, insightful feature engineering, and effective visualization techniques.

## Key Features

- **Data Fetching**: Utilizes the Alpha Vantage API to retrieve historical stock data for various companies.
- **Feature Engineering**:
  - Incorporates moving averages (7-day and 30-day) for trend analysis.
  - Includes lagged closing prices to enhance predictive capabilities.
- **Model Development**:
  - Linear regression model built for its interpretability and simplicity.
  - Robust training on engineered features to capture market trends.
- **Evaluation Metrics**:
  - Employs Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score to assess performance.
- **Visualization**:
  - Plots actual vs. predicted stock prices for intuitive evaluation.
  - Residual analysis to identify discrepancies and improve accuracy.
- **Model Persistence**: Saves the trained model as a `.pkl` file for efficient reuse.

## Performance

The model consistently achieves high accuracy on various stock datasets:
- **R² Score**: Above 0.9, reflecting strong predictive performance.
- **MAE and RMSE**: Indicating minimal deviations between actual and predicted values.

These metrics highlight the robustness of the methodology and the effectiveness of the engineered features.

## Installation and Usage

### Prerequisites

Install the required Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `alpha_vantage`
- `joblib`
- `seaborn`

To install all dependencies, run:

```bash
pip install -r requirements.txt

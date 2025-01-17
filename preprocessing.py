from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def fetch_data(symbol):
    """
    Fetches full historical data for the given stock symbol using the Alpha Vantage API.
    """
    # Initialize Alpha Vantage API key
    api_key = 'QVN9ASG9B5ZV2I80'  # Replace with your API key

    # Fetch data for the symbol provided (e.g., GOOGL)
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')

    # Convert the index to datetime format and sort in ascending order
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)  # Ensure the data is ordered by date

    # Rename columns to consistent names
    data.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    }, inplace=True)

    print(data.head())  # Print the first 5 rows of the DataFrame
    print(data.columns)  # Print the column names
    print(data.index)
    return data


def preprocess_data(data):
    """
    Processes the fetched stock data for training and testing.
    """
    # Add lagged closing prices (Lag1, Lag2, Lag3) to capture historical dependencies
    data['Lag1'] = data['close'].shift(1)  # Replace 'close' with '4. close'
    data['Lag2'] = data['close'].shift(2)
    data['Lag3'] = data['close'].shift(3)
    data['MA7'] = data['close'].rolling(window=7).mean()
    data['MA30'] = data['close'].rolling(window=30).mean()


    # Drop rows with missing values after lagged feature creation
    data.dropna(inplace=True)

    # Separate the features (Lag1, Lag2, etc.) and target (close price)
    X = data[['Lag1', 'Lag2', 'Lag3', 'MA7', 'MA30']]
    y = data['close']

    # Normalize features to scale data into the range [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

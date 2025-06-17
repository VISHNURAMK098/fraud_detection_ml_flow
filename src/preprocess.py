import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame):
    """Encode categorical variables and scale numerical ones."""

    df_balanced = df.copy()

    categorical_cols = [
        'card_type', 'merchant_category', 'transaction_type',
        'device_type', 'entry_method', 'customer_region'
    ]
    df_balanced = pd.get_dummies(df_balanced, columns=categorical_cols, drop_first=True)

    X = df_balanced.drop(
        ["transaction_id", "timestamp", "customer_id", "merchant_id", "is_fraud"], axis=1
    )
    y = df_balanced["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = [
        'amount', 'customer_age', 'hour_of_day',
        'avg_transaction_amt_24h', 'num_prev_transactions_24h'
    ]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler

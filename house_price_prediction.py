import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(7)
n = 500

df = pd.DataFrame({
    'Size': np.random.randint(800, 4000, n),
    'Bedrooms': np.random.randint(1, 6, n),
    'Location': np.random.randint(1, 4, n),  # 1=Rural, 2=Suburb, 3=City
})

df['Price'] = (
    50000
    + df['Size'] * 150
    + df['Bedrooms'] * 10000
    + df['Location'] * 50000
    + np.random.randint(-20000, 20000, n)
)

# randomly drop some Size values to simulate real-world messiness
missing_idx = np.random.choice(df.index, 10, replace=False)
df.loc[missing_idx, 'Size'] = np.nan

print(df.head())
print(df.describe())
print("\nMissing:\n", df.isnull().sum())

df['Price'].plot(kind='hist', bins=30, edgecolor='black')
plt.title('House Price Distribution')
plt.tight_layout()
plt.show()

# TODO: try median here instead and see if it makes a difference
df['Size'].fillna(df['Size'].mean(), inplace=True)

X = df[['Size', 'Bedrooms', 'Location']].copy()
y = df['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=3
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMSE: {mse:.2f}")
print(f"R²:  {r2:.4f}")

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.show()

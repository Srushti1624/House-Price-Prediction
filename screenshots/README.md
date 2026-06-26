## Screenshots

<img width="502" height="386" alt="image" src="https://github.com/user-attachments/assets/a6325fdc-651f-472e-b8d9-35362503fde8" />

<img width="679" height="460" alt="image" src="https://github.com/user-attachments/assets/6a51c885-1b3c-4df3-8861-8576e1452ec1" />

/tmp/ipykernel_674/551405953.py:40: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Size'].fillna(df['Size'].mean(), inplace=True)

MSE: 744835131.82
R²:  0.9589

<img width="654" height="464" alt="image" src="https://github.com/user-attachments/assets/855d040e-a8fd-494b-9d5b-4e0476666a40" />

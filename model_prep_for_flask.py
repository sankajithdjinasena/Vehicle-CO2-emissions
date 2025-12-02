# prepare_data.py (Revised to include 'Model' in feature lists)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# --- 1. Create a dataset and extract features ---
try:
    # Attempt to load actual data
    data = pd.read_csv("co2.csv")
    data.rename(columns={'Make': 'Brand'}, inplace=True)
    
    # Replicate outlier removal/cleaning for feature list accuracy
    numeric_features_raw = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
                        'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
                        'Fuel Consumption Comb (mpg)']
    def remove_outliers_iqr(df, columns):
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean
    df = remove_outliers_iqr(data, numeric_features_raw)
except FileNotFoundError:
    # Fallback/Dummy Data
    print("Warning: co2.csv not found. Using dummy data for model saving.")
    data = {'Brand': ['ACURA', 'FORD', 'TESLA', 'TESLA', 'FORD'],
            'Model': ['ILX', 'F-150', 'Model S', 'Model X', 'MUSTANG'],
            'Vehicle Class': ['COMPACT', 'SUV - SMALL', 'COMPACT', 'SUV - SMALL', 'COMPACT'],
            'Engine Size(L)': [2.0, 3.5, 0.0, 0.0, 5.0],
            'Cylinders': [4, 6, 0, 0, 8],
            'Transmission': ['AS5', 'A6', 'A1', 'A1', 'M6'],
            'Fuel Type': ['Z', 'X', 'X', 'X', 'Z'],
            'Fuel Consumption City (L/100 km)': [9.9, 13.0, 0.0, 0.0, 15.0],
            'Fuel Consumption Hwy (L/100 km)': [6.7, 9.7, 0.0, 0.0, 10.0],
            'Fuel Consumption Comb (L/100 km)': [8.5, 11.5, 0.0, 0.0, 13.0],
            'Fuel Consumption Comb (mpg)': [33, 25, 0, 0, 22],
            'CO2 Emissions(g/km)': [196, 264, 0, 0, 300]}
    df = pd.DataFrame(data)
    df.rename(columns={'Make': 'Brand'}, inplace=True)

# --- Define Categorical Features for Selection (NOW INCLUDING 'Model') ---
CAT_FEATURES = ['Brand', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

# --- 2. Extract and Save Unique Feature Lists ---
feature_lists = {}
for col in CAT_FEATURES:
    # Get sorted unique values (this ensures ALL unique values are captured)
    feature_lists[col] = sorted(df[col].unique().tolist())

joblib.dump(feature_lists, 'feature_lists.pkl')
print("Unique categorical feature lists saved as feature_lists.pkl (including Model).")

# --- 3. Save the Model (Same as before) ---
numeric_features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
                    'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
                    'Fuel Consumption Comb (mpg)']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, CAT_FEATURES) # Use all categorical features
    ],
    remainder='drop'
)

dt_model = DecisionTreeRegressor(random_state=42)
clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', dt_model)])

X = df.drop('CO2 Emissions(g/km)', axis=1)
y = df['CO2 Emissions(g/km)']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, 'best_model.pkl')
print("Model pipeline saved as best_model.pkl.")
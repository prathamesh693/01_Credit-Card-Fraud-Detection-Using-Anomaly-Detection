import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("02_Data/creditcard.csv")

# Fill missing values if any
df.fillna(df.median(), inplace=True)

# Scale 'Amount' and 'Time'
scaler = RobustScaler()
df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

# Separate features and target
X = df.drop(columns='Class')
y = df['Class']

# MinMax scaling for features
minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X)

# Optional: PCA dimensionality reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Convert back to DataFrame
X_pca_df = pd.DataFrame(X_pca)
X_pca_df['Class'] = y.values

# Split into train and test (e.g., 80-20 split)
X_train, X_test = train_test_split(X_pca_df, test_size=0.2, random_state=42, stratify=X_pca_df['Class'])

# Save files
X_train.to_csv("6_Output/train_preprocessed.csv", index=False)
X_test.to_csv("6_Output/test_preprocessed.csv", index=False)

print("Preprocessing complete.")
print("Saved train data to output/train_preprocessed.csv")
print("Saved test data to output/test_preprocessed.csv")

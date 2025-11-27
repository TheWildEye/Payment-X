# train_nn.py — requires tensorflow installed
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

DATA_PATH = "C:\Users\vyomn\Downloads\AI\payment_fraud.csv"
df = pd.read_csv(DATA_PATH)
target_col = df.columns[-1]
X = df.drop(columns=[target_col]); y = df[target_col]

# encode target if needed
if y.dtype == object or y.dtype.name == "category":
    le = LabelEncoder(); y = le.fit_transform(y); joblib.dump(le, "label_encoder.pkl")

# impute
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
if num_cols:
    num_imp = SimpleImputer(strategy="median").fit(X[num_cols]); X[num_cols] = num_imp.transform(X[num_cols])
if cat_cols:
    cat_imp = SimpleImputer(strategy="most_frequent").fit(X[cat_cols]); X[cat_cols] = cat_imp.transform(X[cat_cols])
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(len(set(y)), activation="softmax" if len(set(y))>2 else "sigmoid")
])
loss = "sparse_categorical_crossentropy" if len(set(y))>2 else "binary_crossentropy"
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)
model.save("nn_model.h5")
joblib.dump(scaler, "scaler.pkl")

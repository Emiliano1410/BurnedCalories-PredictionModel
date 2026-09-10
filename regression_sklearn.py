import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Extraccion
df = pd.read_csv("gym_members_exercise_tracking.csv")

# 2. Transformacion
# One-hot encoding de variables categorica
df = pd.get_dummies(df, columns=["Gender"], drop_first=False)

# Features seleccionadas según el analisis de correlacion del modelo from scratch
features = [
    "Age",
    "Gender_Male",
    "Avg_BPM",
    "Session_Duration (hours)",
    "Fat_Percentage",
    "Water_Intake (liters)",
    "Experience_Level",
]
target = "Calories_Burned"

X = df[features].astype(float)
y = df[target].astype(float)

# 3. Division del dataset: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 4. Normalizacion min-max
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# 5. Entrenamiento del modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train_scaled, y_train_scaled)

# 6. Predicciones
pred_train = modelo.predict(X_train_scaled)
pred_val = modelo.predict(X_val_scaled)
pred_test = modelo.predict(X_test_scaled)

# 7. Metricas
resultados = pd.DataFrame({
    "Conjunto": ["Train", "Validation", "Test"],
    "MSE normalizado": [
        mean_squared_error(y_train_scaled, pred_train),
        mean_squared_error(y_val_scaled, pred_val),
        mean_squared_error(y_test_scaled, pred_test),
    ],
    "R2": [
        r2_score(y_train_scaled, pred_train),
        r2_score(y_val_scaled, pred_val),
        r2_score(y_test_scaled, pred_test),
    ],
})
print(resultados.to_string(index=False))

# 8. Importancia de las features para el entrenamiento del modelo
print("\nImportancia de las features:")
for nombre, importancia in sorted(
    # Asociar cada feature con su importancia de menor a mayor
    zip(features, modelo.feature_importances_), key=lambda x: -x[1]
):
    print(f"{nombre}: {importancia:.4f}")

# 9. Grafica: valores reales vs predicciones (test set)
plt.figure(figsize=(8, 5))
indices = np.arange(len(y_test_scaled))
plt.scatter(indices, y_test_scaled, label="Valores reales", alpha=0.7)
plt.scatter(indices, pred_test, label="Predicciones", alpha=0.7)
plt.title("Valores reales vs predicciones (Test set) - Random Forest (scikit-learn)")
plt.xlabel("Índice de muestra")
plt.ylabel("Calorías quemadas (normalizado)")
plt.legend()
plt.tight_layout()
plt.savefig("reales_vs_predicciones_sklearn.png", dpi=150)
plt.close()

# 10. Grafica: distribución de residuos (test set)
residuos = y_test_scaled - pred_test
plt.figure(figsize=(8, 5))
plt.hist(residuos, bins=20)
plt.title("Distribución de residuos (Test set) - Random Forest (scikit-learn)")
plt.xlabel("Residuo (real - predicho)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("distribucion_residuos_sklearn.png", dpi=150)
plt.close()
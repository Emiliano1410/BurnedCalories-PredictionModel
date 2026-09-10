import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Funciones del modelo
__errors_train__ = []
__errors_val__ = []


def h(params, sample):
    """Hipotesis lineal: h(x) = a + b*x1 + c*x2 + ... """
    acum = 0
    for i in range(len(params)):
        #Tomo los parametros actuales y una muestra
        acum = acum + params[i] * sample[i]
    return acum


def mse(params, samples, y):
    """Calcula el Mean Squared Error para un conjunto de datos."""
    error_acum = 0
    for i in range(len(samples)):
        # Calcula la prediccion
        hyp = h(params, samples[i])
        # Calcula el error
        error = hyp - y[i]
        error_acum += error ** 2
    return error_acum / len(samples)


def GD(params, samples, y, lr):
    """Un paso de descenso de gradiente sobre todo el set de entrenamiento."""
    # Actualizar con los parametros originales de cada iteracion
    temp = list(params)
    
    for j in range(len(params)):
        acum = 0
        for i in range(len(samples)):
            error = h(params, samples[i]) - y[i]
            acum = acum + error * samples[i][j]
        temp[j] = params[j] - lr * (1 / len(samples)) * acum
    return temp


def r2_score(params, samples, y):
    """Calcula R^2"""
    y_mean = sum(y) / len(y)
    ss_res = 0
    ss_tot = 0
    for i in range(len(samples)):
        pred = h(params, samples[i])
        ss_res += (y[i] - pred) ** 2
        ss_tot += (y[i] - y_mean) ** 2
    return 1 - (ss_res / ss_tot)

# 2. Carga y preparacion de los datos
df = pd.read_csv("gym_members_exercise_tracking.csv")

# Codificacion de variables categoricas con one hot encoding
df = pd.get_dummies(df, columns=["Gender", "Workout_Type"], drop_first=False)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# Variable objetivo
y_col = "Calories_Burned"

# Excluir la variable objetivo de mis features
feature_cols = [c for c in df.columns if c != y_col]

# 2.1 Matriz de correlacion de los features
corr_matrix = df[feature_cols + [y_col]].corr()
 
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Coeficiente de correlacion")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Matriz de correlacion - Features y variable objetivo")
plt.tight_layout()
plt.savefig("scratch_correlation_matrix.png")
plt.show()
 
print("\nMatriz de correlacion:")
print(corr_matrix.round(2))
 
# Correlacion de cada feature con la variable objetivo
print("\nCorrelacion de cada feature con", y_col, "(ordenado):")
print(corr_matrix[y_col].drop(y_col).sort_values(ascending=False).round(3))

# 2.2 Eliminacion de features con base en el analisis de correlacion
columns_to_drop = [
    "Weight (kg)",
    "Height (m)",
    "Max_BPM",
    "Resting_BPM",
    "BMI",
    "Workout_Frequency (days/week)",
    "Workout_Type_Cardio",
    "Workout_Type_HIIT",
    "Workout_Type_Strength",
    "Workout_Type_Yoga",
    "Gender_Female"
]
 
df = df.drop(columns=columns_to_drop)
feature_cols = [c for c in feature_cols if c not in columns_to_drop]

# Normalizacion min-max de todas las columnas numericas
df_norm = df.copy()
norm_params = {}
for col in feature_cols + [y_col]:
    col_min = df[col].min()
    col_max = df[col].max()
    norm_params[col] = (col_min, col_max)
    df_norm[col] = (df[col] - col_min) / (col_max - col_min)

# Features
X = df_norm[feature_cols].values.tolist()
# Objetivo
y = df_norm[y_col].values.tolist()
# Agrego el bias
X = [[1] + row for row in X]

# 3. Division en train (60%), validation (20%) y test (20%)
np.random.seed(42)
indices = list(range(len(X)))
np.random.shuffle(indices)

n = len(X)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
X_val = [X[i] for i in val_idx]
y_val = [y[i] for i in val_idx]
X_test = [X[i] for i in test_idx]
y_test = [y[i] for i in test_idx]

# 4. Entrenamiento con descenso de gradiente
params = [0.0] * len(X_train[0])  # inicializacion en 0 de los parametros
lr = 0.3
epochs = 0
max_epochs = 10000

while True:
    old_params = list(params)
    params = GD(params, X_train, y_train, lr)

    train_error = mse(params, X_train, y_train)
    val_error = mse(params, X_val, y_val)
    __errors_train__.append(train_error)
    __errors_val__.append(val_error)

    epochs += 1
    if old_params == params or epochs == max_epochs:
        break

print("Epochs corridas:", epochs)
print("\nParametros finales:")
print(f"Bias: {params[0]:.4f}")

for feature, parameter in zip(feature_cols, params[1:]):
    print(f"{feature}: {parameter:.4f}")

# 5. Calcular mse y r2 para cada split
train_mse = mse(params, X_train, y_train)
val_mse = mse(params, X_val, y_val)
test_mse = mse(params, X_test, y_test)

train_r2 = r2_score(params, X_train, y_train)
val_r2 = r2_score(params, X_val, y_val)
test_r2 = r2_score(params, X_test, y_test)

print(f"Train MSE: {train_mse:.4f}, Train R2: {train_r2:.4f}")
print(f"Validation MSE: {val_mse:.4f}, Validation R2: {val_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

# 6. Graficas
# Grafica 1: Curva de error de train y validation por epoch
plt.figure()
plt.plot(__errors_train__, label="Train MSE")
plt.plot(__errors_val__, label="Validation MSE")
plt.title("MSE de Train y Validation a lo largo de las epocas")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# Grafica 2: Valores reales vs predichos
predictions_test = [h(params, sample) for sample in X_test]
plt.figure()
plt.scatter(range(len(y_test)), y_test, label="Valores reales", alpha=0.6)
plt.scatter(range(len(predictions_test)), predictions_test, label="Predicciones", alpha=0.6)
plt.title("Valores reales vs predicciones (Test set) - Modelo desde cero")
plt.xlabel("Indice de muestra")
plt.ylabel("Calorias quemadas (normalizado)")
plt.legend()
plt.savefig("true_vs_pred.png")
plt.show()

# Grafica 3: Distribucion de residuos
residuals = [y_test[i] - predictions_test[i] for i in range(len(y_test))]
plt.figure()
plt.hist(residuals, bins=30)
plt.title("Distribucion de residuos (Test set) - Modelo desde cero")
plt.xlabel("Residuo (real - predicho)")
plt.ylabel("Frecuencia")
plt.savefig("residuals.png")
plt.show()
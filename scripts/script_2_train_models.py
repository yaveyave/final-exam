"""
SCRIPT 2: Train Models - 4 Algoritmos ML
Entrenamiento con validación completa: GridSearchCV, K-Fold CV, Matriz de Confusión
Métricas: Accuracy, Precision, Recall, F1, R², RMSE, MAE

Ejecutar desde: scripts/script_2_train_models.py
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 2: ENTRENAMIENTO DE MODELOS ML CON VALIDACIÓN COMPLETA")
print("=" * 80)

# Cargar datos
df = pd.read_csv('data/processed/tsp_dataset.csv')
print(f"\nDataset cargado: {len(df)} instancias, {df.shape[1]} columnas")

# Preparar características y targets
X = df[['n_cities', 'avg_distance', 'std_distance', 'max_distance', 'cv_distance']].values
y_reg = df['log_time'].values
y_class = df['difficulty_class'].values

# Mapear clases a números
class_mapping = {'Easy': 0, 'Medium': 1, 'Hard': 2}
y_class_numeric = np.array([class_mapping[c] for c in y_class])

# Dividir datos
X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class = train_test_split(
    X, y_reg, y_class_numeric, test_size=0.2, random_state=42, stratify=y_class_numeric
)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Conjunto de entrenamiento: {len(X_train)} instancias")
print(f"Conjunto de prueba: {len(X_test)} instancias")
print("\n" + "=" * 80)
print("ENTRENAMIENTO DE MODELOS")
print("=" * 80)

# ===== MODELO 1: kNN CON BÚSQUEDA DE MEJOR K Y K-FOLD =====
print("\n[1/4] kNN - Optimización de hiperparámetro k...", flush=True)

param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15]}
knn_base = KNeighborsClassifier()
grid_search = GridSearchCV(knn_base, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train_class)

best_k = grid_search.best_params_['n_neighbors']
best_cv_score = grid_search.best_score_

print(f"  - Mejor k encontrado: {best_k}")
print(f"  - CV Score (GridSearchCV): {best_cv_score:.4f}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train_class)

# K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(knn, X_train_scaled, y_train_class, cv=skf, scoring='accuracy')
print(f"  - K-Fold CV (k={best_k}): {kfold_scores.mean():.4f} (+/- {kfold_scores.std():.4f})")

y_pred_knn = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test_class, y_pred_knn)
knn_precision = precision_score(y_test_class, y_pred_knn, average='weighted', zero_division=0)
knn_recall = recall_score(y_test_class, y_pred_knn, average='weighted', zero_division=0)
knn_f1 = f1_score(y_test_class, y_pred_knn, average='weighted', zero_division=0)

joblib.dump(knn, 'results/models/model_knn.pkl')

print(f"  - Accuracy (test): {knn_accuracy:.4f}")
print(f"  - Precision (weighted): {knn_precision:.4f}")
print(f"  - Recall (weighted): {knn_recall:.4f}")
print(f"  - F1-Score (weighted): {knn_f1:.4f}")

# Matriz de Confusión
cm_knn = confusion_matrix(y_test_class, y_pred_knn)
print("\n  Matriz de Confusión (kNN):")
print(f"    {cm_knn}")

# Reporte por clase
print("\n  Reporte por clase (Easy/Medium/Hard):")
class_names = ['Easy', 'Medium', 'Hard']
print(classification_report(y_test_class, y_pred_knn, target_names=class_names, digits=4))

# Guardar matriz de confusión
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm_knn, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues')
plt.title('Matriz de Confusión - kNN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/confusion_matrix_knn.png', dpi=100)
print("  ✓ Matriz de confusión guardada: results/plots/confusion_matrix_knn.png")
plt.close()

# ===== MODELO 2: REGRESIÓN LINEAL =====
print("\n[2/4] Regresión Lineal...", flush=True)

lineal = LinearRegression()
lineal.fit(X_train_scaled, y_train_reg)

# K-Fold CV para regresión
kfold_r2_lineal = cross_val_score(lineal, X_train_scaled, y_train_reg, cv=5, scoring='r2')
print(f"  - K-Fold CV R²: {kfold_r2_lineal.mean():.4f} (+/- {kfold_r2_lineal.std():.4f})")

y_pred_lineal = lineal.predict(X_test_scaled)
lineal_r2 = lineal.score(X_test_scaled, y_test_reg)
lineal_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_lineal))
lineal_mae = mean_absolute_error(y_test_reg, y_pred_lineal)

joblib.dump(lineal, 'results/models/model_lineal.pkl')

print(f"  - R² (test): {lineal_r2:.4f}")
print(f"  - RMSE: {lineal_rmse:.4f}")
print(f"  - MAE: {lineal_mae:.4f}")

# ===== MODELO 3: RANDOM FOREST =====
print("\n[3/4] Random Forest (n_estimators=100)...", flush=True)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_reg)

# K-Fold CV para RF
kfold_r2_rf = cross_val_score(rf, X_train_scaled, y_train_reg, cv=5, scoring='r2')
print(f"  - K-Fold CV R²: {kfold_r2_rf.mean():.4f} (+/- {kfold_r2_rf.std():.4f})")

y_pred_rf = rf.predict(X_test_scaled)
rf_r2 = rf.score(X_test_scaled, y_test_reg)
rf_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
rf_mae = mean_absolute_error(y_test_reg, y_pred_rf)

joblib.dump(rf, 'results/models/model_rf.pkl')

print(f"  - R² (test): {rf_r2:.4f}")
print(f"  - RMSE: {rf_rmse:.4f}")
print(f"  - MAE: {rf_mae:.4f}")

# Feature importance
feature_names = ['n_cities', 'avg_distance', 'std_distance', 'max_distance', 'cv_distance']
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
importance_df.to_csv('results/analysis/feature_importance.csv', index=False)

print("\n  Feature Importance:")
for idx, row in importance_df.iterrows():
    print(f"    {row['Feature']}: {row['Importance']:.4f}")

# ===== MODELO 4: MLP KERAS =====
print("\n[4/4] MLP Keras (Dense 32→16→1)...", flush=True)

mlp = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dense(16, activation='relu'),
    Dense(1)
])
mlp.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = mlp.fit(X_train_scaled, y_train_reg, 
                  epochs=100, 
                  batch_size=4,
                  validation_split=0.2, 
                  callbacks=[early_stop], 
                  verbose=0)

mlp_mse = mlp.evaluate(X_test_scaled, y_test_reg, verbose=0)[0]
y_pred_mlp = mlp.predict(X_test_scaled, verbose=0).flatten()
mlp_rmse = np.sqrt(mlp_mse)
mlp_mae = mean_absolute_error(y_test_reg, y_pred_mlp)

# Calcular R² para MLP
ss_res = np.sum((y_test_reg - y_pred_mlp) ** 2)
ss_tot = np.sum((y_test_reg - np.mean(y_test_reg)) ** 2)
mlp_r2 = 1 - (ss_res / ss_tot)

mlp.save('results/models/model_mlp.keras')

print(f"  - R² (test): {mlp_r2:.4f}")
print(f"  - MSE: {mlp_mse:.4f}")
print(f"  - RMSE: {mlp_rmse:.4f}")
print(f"  - MAE: {mlp_mae:.4f}")
print(f"  - Epochs (early stop): {len(history.history['loss'])}")

# Guardar curva de aprendizaje (loss)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss (MSE)', fontsize=11)
ax1.set_title('Learning Curve - MLP (Loss)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('MAE', fontsize=11)
ax2.set_title('Learning Curve - MLP (MAE)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/mlp_learning_curves.png', dpi=100)
print("  ✓ Learning curves guardadas: results/plots/mlp_learning_curves.png")
plt.close()

# ===== RESUMEN FINAL =====
print("\n" + "=" * 80)
print("RESUMEN DE MODELOS - 145 INSTANCIAS")
print("=" * 80)

summary_data = {
    'Modelo': ['kNN', 'Lineal', 'Random Forest', 'MLP'],
    'Métrica': [f'{knn_accuracy:.4f}', f'{lineal_r2:.4f}', f'{rf_r2:.4f}', f'{mlp_r2:.4f}'],
    'Tipo': ['Accuracy', 'R²', 'R²', 'R²'],
    'Tarea': ['Clasificación', 'Regresión', 'Regresión', 'Regresión']
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "-" * 80)
print("MÉTRICAS DETALLADAS POR MODELO")
print("-" * 80)

print("\nkNN (Clasificación):")
print(f"  Accuracy:           {knn_accuracy:.4f}")
print(f"  Precision (weighted): {knn_precision:.4f}")
print(f"  Recall (weighted):    {knn_recall:.4f}")
print(f"  F1-Score (weighted):  {knn_f1:.4f}")
print(f"  Mejor k:             {best_k}")
print(f"  GridSearchCV CV Score: {best_cv_score:.4f}")
print(f"  K-Fold CV (5-split):  {kfold_scores.mean():.4f} (+/- {kfold_scores.std():.4f})")

print("\nRegresión Lineal:")
print(f"  R² (test):          {lineal_r2:.4f}")
print(f"  K-Fold CV R²:       {kfold_r2_lineal.mean():.4f} (+/- {kfold_r2_lineal.std():.4f})")
print(f"  RMSE:               {lineal_rmse:.4f}")
print(f"  MAE:                {lineal_mae:.4f}")

print("\nRandom Forest:")
print(f"  R² (test):          {rf_r2:.4f}")
print(f"  K-Fold CV R²:       {kfold_r2_rf.mean():.4f} (+/- {kfold_r2_rf.std():.4f})")
print(f"  RMSE:               {rf_rmse:.4f}")
print(f"  MAE:                {rf_mae:.4f}")

print("\nMLP Keras:")
print(f"  R² (test):          {mlp_r2:.4f}")
print(f"  RMSE:               {mlp_rmse:.4f}")
print(f"  MAE:                {mlp_mae:.4f}")
print(f"  MSE:                {mlp_mse:.4f}")
print(f"  Epochs (early stop): {len(history.history['loss'])}")

# Guardar resumen detallado
detailed_summary = pd.DataFrame({
    'Modelo': ['kNN', 'Lineal', 'Random Forest', 'MLP'],
    'Accuracy/R²': [f'{knn_accuracy:.4f}', f'{lineal_r2:.4f}', f'{rf_r2:.4f}', f'{mlp_r2:.4f}'],
    'K-Fold CV': [f'{kfold_scores.mean():.4f}±{kfold_scores.std():.4f}',
                  f'{kfold_r2_lineal.mean():.4f}±{kfold_r2_lineal.std():.4f}',
                  f'{kfold_r2_rf.mean():.4f}±{kfold_r2_rf.std():.4f}',
                  'N/A'],
    'RMSE': ['-', f'{lineal_rmse:.4f}', f'{rf_rmse:.4f}', f'{mlp_rmse:.4f}'],
    'MAE': ['-', f'{lineal_mae:.4f}', f'{rf_mae:.4f}', f'{mlp_mae:.4f}'],
    'Notas': [f'k={best_k}', 'Baseline', 'n_est=100', f'Ep={len(history.history["loss"])}']
})
detailed_summary.to_csv('results/analysis/models_summary.csv', index=False)

print("\n" + "=" * 80)
print("✓ Modelos entrenados y guardados en results/models/")
print("✓ Análisis guardado en results/analysis/")
print("✓ Visualizaciones guardadas en results/plots/")
print("=" * 80)

joblib.dump(scaler, 'results/models/scaler.pkl')
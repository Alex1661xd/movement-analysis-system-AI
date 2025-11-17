"""
Script para reentrenar SVM con probability=True
Ejecutar en Google Colab AL FINAL del notebook de Entrega 3
"""

import joblib
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

print("="*80)
print("🔄 REENTRENANDO SVM CON PROBABILIDADES")
print("="*80)

# Configuración de rutas
RESULTADOS_PATH_E3 = "/content/drive/MyDrive/APO3_EntregaFinal/Entrega3/resultados"

# Grid de hiperparámetros (igual que antes)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

print("\n🤖 Reentrenando SVM con probability=True...")
print("   (Esto puede tomar un poco más de tiempo)")

# Entrenar con probability=True
grid_svm_prob = GridSearchCV(
    SVC(random_state=42, probability=True),  # ← probability=True
    param_grid_svm,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)
grid_svm_prob.fit(X_pca_train, y_pca_train)
best_svm_prob = grid_svm_prob.best_estimator_

print(f"\n✅ Mejores hiperparámetros: {grid_svm_prob.best_params_}")
print(f"📊 Mejor CV score: {grid_svm_prob.best_score_*100:.2f}%")

# Evaluar en test
from sklearn.metrics import accuracy_score
y_pred_test = best_svm_prob.predict(X_pca_test)
acc_test = accuracy_score(y_pca_test, y_pred_test)
print(f"📈 Test accuracy: {acc_test*100:.2f}%")

# Verificar que ahora sí tiene probabilidades
print("\n🔍 Verificando predict_proba...")
try:
    proba_test = best_svm_prob.predict_proba(X_pca_test[:1])
    print(f"   ✅ predict_proba funciona correctamente!")
    print(f"   Probabilidades de ejemplo: {proba_test[0][:3]}")
except:
    print(f"   ❌ Error: predict_proba no disponible")

# Guardar modelo actualizado
print("\n💾 Guardando modelo SVM con probabilidades...")
joblib.dump(best_svm_prob, os.path.join(RESULTADOS_PATH_E3, "svm_model.pkl"))
print(f"   ✅ Guardado en: {RESULTADOS_PATH_E3}/svm_model.pkl")

print("\n" + "="*80)
print("🎉 ¡SVM reentrenado con probabilidades!")
print("="*80)
print("\n📝 Siguiente paso:")
print("   1. Descarga el archivo 'svm_model.pkl' de Google Drive")
print("   2. Reemplázalo en tu carpeta local Entrega3/resultados/")
print("   3. Reinicia la aplicación Streamlit")


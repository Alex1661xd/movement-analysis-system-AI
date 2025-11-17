"""
Script para guardar los modelos entrenados de Entrega 3
Ejecutar este código AL FINAL del entrenamiento en Google Colab
"""

import joblib
import os

# Configurar ruta
RESULTADOS_PATH_E3 = "/content/drive/MyDrive/APO3_EntregaFinal/Entrega3/resultados"

# Guardar los 3 modelos entrenados
print("💾 Guardando modelos entrenados...")

# 1. Random Forest
joblib.dump(best_rf_e3, os.path.join(RESULTADOS_PATH_E3, "random_forest_model.pkl"))
print("   ✅ Random Forest guardado")

# 2. SVM
joblib.dump(best_svm_e3, os.path.join(RESULTADOS_PATH_E3, "svm_model.pkl"))
print("   ✅ SVM guardado")

# 3. XGBoost
joblib.dump(best_xgb_e3, os.path.join(RESULTADOS_PATH_E3, "xgboost_model.pkl"))
print("   ✅ XGBoost guardado")

# 4. Label Encoder (para convertir predicciones numéricas a nombres)
joblib.dump(label_encoder_e3, os.path.join(RESULTADOS_PATH_E3, "label_encoder.pkl"))
print("   ✅ Label Encoder guardado")

print("\n🎉 Todos los modelos guardados correctamente!")
print(f"📁 Ubicación: {RESULTADOS_PATH_E3}")


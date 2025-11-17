# 🏃 Aplicación de Clasificación de Movimiento Humano

Aplicación web interactiva para clasificar actividades humanas usando visión por computadora y machine learning.

---

## 🎯 Características

- **Carga de videos** en formatos MP4, AVI, MOV
- **Extracción automática de landmarks** con MediaPipe Pose
- **Clasificación con 3 modelos** de Machine Learning:
  - Random Forest (94.44% accuracy)
  - SVM RBF (94.44% accuracy)
  - XGBoost (94.44% accuracy)
- **Visualización de métricas biomecánicas** en tiempo real
- **Análisis PCA** de componentes principales
- **Descarga de resultados** en formato CSV

---

## 🚀 Instalación

### Paso 1: Clonar el repositorio

```bash
cd Entrega3
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3: Asegurar que los modelos estén presentes

Antes de ejecutar la aplicación, debes tener estos archivos en la carpeta `resultados/`:

```
resultados/
├── random_forest_model.pkl
├── svm_model.pkl
├── xgboost_model.pkl
├── pca_model.pkl
├── scaler_minmax.pkl
└── label_encoder.pkl
```

**⚠️ Importante:** Si aún no has guardado los modelos, ejecuta en Google Colab:

```python
# Al final de tu notebook de Entrega 3, agregar:
import joblib
import os

RESULTADOS_PATH = "/content/drive/MyDrive/APO3_EntregaFinal/Entrega3/resultados"

joblib.dump(best_rf_e3, os.path.join(RESULTADOS_PATH, "random_forest_model.pkl"))
joblib.dump(best_svm_e3, os.path.join(RESULTADOS_PATH, "svm_model.pkl"))
joblib.dump(best_xgb_e3, os.path.join(RESULTADOS_PATH, "xgboost_model.pkl"))
joblib.dump(label_encoder_e3, os.path.join(RESULTADOS_PATH, "label_encoder.pkl"))

print("✅ Modelos guardados!")
```

Luego descarga los archivos `.pkl` de Google Drive y colócalos en tu carpeta `Entrega3/resultados/`.

---

## ▶️ Ejecutar la Aplicación

```bash
streamlit run app_clasificador_movimiento.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 📖 Cómo usar

1. **Abre la aplicación** en tu navegador
2. **Sube un video** usando el botón de carga
3. **Haz clic en "Analizar Video"**
4. **Revisa los resultados:**
   - Predicción final por consenso de modelos
   - Predicciones individuales de cada modelo
   - Métricas biomecánicas extraídas
   - Análisis de componentes principales
5. **Descarga los resultados** en CSV si lo deseas

---

## 🎬 Actividades que puede clasificar

1. **Adelante** - Caminar hacia la cámara
2. **Atrás** - Caminar alejándose
3. **Sentado** - Posición sentada
4. **Cadera al frente** - Flexión de cadera frontal
5. **Caderas** - Rotación de caderas
6. **Lado** - Movimiento lateral
7. **Sentadilla** - Sentadilla profunda
8. **Tijeras** - Movimiento de tijeras

---

## 📊 Tecnologías Utilizadas

- **MediaPipe Pose** - Detección de landmarks corporales
- **Scikit-learn** - Random Forest y SVM
- **XGBoost** - Gradient Boosting
- **PCA** - Reducción dimensional (12 → 6 características)
- **Streamlit** - Interfaz web interactiva
- **OpenCV** - Procesamiento de video

---

## 🔧 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'streamlit'"

```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: [Errno 2] No such file or directory: 'resultados/random_forest_model.pkl'"

Asegúrate de tener todos los archivos `.pkl` en la carpeta `resultados/`. Ejecuta `guardar_modelos.py` en Colab primero.

### Error: "No se pudo procesar el video"

- Verifica que el video tenga una persona visible de cuerpo completo
- Asegúrate que la iluminación sea adecuada
- Prueba con un video más corto (2-5 segundos)

### La aplicación no se abre automáticamente

Abre manualmente en tu navegador: `http://localhost:8501`

---

## 📝 Notas Importantes

- Los modelos fueron entrenados con **86 videos** de 8 categorías
- El sistema funciona mejor con:
  - Videos de 2-5 segundos
  - Persona visible de cuerpo completo
  - Buena iluminación
  - Fondo sin mucho movimiento

- **Limitaciones:**
  - Dataset pequeño limita generalización
  - Mejor rendimiento con personas similares al conjunto de entrenamiento
  - Puede tener dificultades con ángulos de cámara muy diferentes

---

## 👥 Equipo de Desarrollo

**Proyecto Final APO 3 - Entrega 3**

- Mariana De La Cruz - A00399618
- Valentina Gómez - A00398790
- Alexis Delgado - A00399176
- Juan Camilo Amorocho - A00399789

---

## 📜 Licencia

Este proyecto es parte del curso APO 3 y es de uso académico.

---

## 🎓 Referencias

- [MediaPipe Pose Documentation](https://google.github.io/mediapipe/solutions/pose.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)


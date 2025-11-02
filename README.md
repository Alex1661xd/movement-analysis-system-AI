# **Entrega 1 y 2 - Proyecto Final APO 3**

## **Sistema de Anotación de Video para Análisis de Actividades Humanas**

**Integrantes del grupo:**

* Mariana De La Cruz - A00399618
* Valentina Gómez - A00398790
* Alexis Delgado - A00399176
* Juan Camilo Amorocho - A00399789



### **Descripción del proyecto**

El repositorio **`movement-analysis-system-IA`** contiene el desarrollo completo del proyecto final del curso **APO 3**, cuyo objetivo es construir un sistema automatizado para el análisis y clasificación de actividades humanas a partir de video, integrando visión por computadora, aprendizaje automático y análisis biomecánico.

El sistema utiliza **MediaPipe Pose**, que permite identificar 33 puntos de referencia corporales (landmarks). A partir de estos, se extraen métricas como brillo, movimiento, velocidad de cadera, ángulos articulares e inclinación de hombros, con el fin de evaluar la postura y clasificar acciones básicas como caminar, sentarse, agacharse o girar.

Estas métricas son utilizadas por modelos de **Machine Learning (Random Forest, SVM y XGBoost)** para clasificar distintas posturas y acciones humanas, logrando un sistema capaz de detectar automáticamente el tipo de movimiento a partir de la información biomecánica derivada del video.



## **Estructura del repositorio**

```
movement-analysis-system-IA/
│
├── README.md                      → Descripción general del proyecto
│
├── APO3_EntregaFinal/
│   ├── Entrega1/                  → Fase inicial del proyecto
│   │   ├── videos/                → Videos originales por categoría
│   │   ├── procesados/            → Videos con esqueleto superpuesto
│   │   ├── landmarks/             → Coordenadas corporales (CSV)
│   │   └── resultados/            → Métricas y reportes sin landmarks
│   │
│   └── Entrega2/                  → Fase de modelado y entrenamiento
│       ├── videos/                → Nuevos videos de entrenamiento
│       ├── procesados/            → Visualización de poses detectadas
│       ├── landmarks/             → Landmarks extraídos (33 joints)
│       └── resultados/            → Datasets limpios, normalizados y métricas
│
├── Entrega 1/
│   └── Entrega1_ProyectoFinal_APO3_MovementAnalysis.ipynb
│
└── Entrega 2/
    └── Entrega2_ProyectoFinal_APO3_MovementAnalysis.ipynb
```

---

## **Fases del proyecto**

### **Entrega 1 — Procesamiento y análisis inicial**

El notebook `Entrega1_ProyectoFinal_APO3_MovementAnalysis.ipynb` incluye:

* Contexto, objetivos y metodología del proyecto.
* Extracción de métricas visuales sin landmarks (brillo, movimiento, duración, FPS).
* Implementación inicial de **MediaPipe Pose** para detección corporal.
* Generación de reportes y métricas descriptivas por categoría de acción.
* Análisis exploratorio básico (EDA) y visualizaciones comparativas.
* Reflexión ética sobre el uso responsable de la visión por computadora.

Resultados disponibles en:
`APO3_EntregaFinal/Entrega1/resultados/`

---

### **Entrega 2 — Normalización, modelado y clasificación**

El notebook `Entrega2_ProyectoFinal_APO3_MovementAnalysis.ipynb` profundiza en la segunda etapa del proyecto, centrada en la creación del modelo de clasificación inteligente.

Incluye:

1. **Estrategia de ampliación de datos:** incorporación de nuevas categorías y ángulos (caderas, lado, sentadillas, tijeras).
2. **Preparación del dataset:**

   * Limpieza de datos y eliminación de columnas irrelevantes (`video`, `resolución`, `fps`).
   * Detección y manejo de outliers.
   * Normalización con **MinMaxScaler**.
3. **Análisis estadístico y correlacional:**

   * Matriz de correlación y visualización con mapa de calor (`sns.heatmap`).
   * Análisis de distribución y relación entre métricas biomecánicas.
4. **Entrenamiento de modelos:**

   * Implementación de **Random Forest**, **SVM (RBF)** y **XGBoost**.
   * Ajuste de hiperparámetros con **GridSearchCV**.
5. **Evaluación comparativa:**

   * Métricas de *accuracy*, *precision*, *recall* y *F1-score*.
   * Visualización de matrices de confusión.
   * Comparación gráfica del rendimiento de cada modelo.
6. **Exportación de resultados y modelos:**

   * Guardado de datasets (`dataset_limpio.csv`, `dataset_normalizado.csv`).

Resultados disponibles en:
`APO3_EntregaFinal/Entrega2/resultados/`

---

### **Organización general de las entregas**

La carpeta principal **`APO3_EntregaFinal`** contiene las dos fases del proyecto:

📂 **videos/** — grabaciones originales realizadas con cámara RGB (teléfono móvil).
📂 **procesados/** — videos con el esqueleto 3D superpuesto y análisis visual de pose.
📂 **landmarks/** — archivos CSV con las coordenadas de las 33 articulaciones detectadas por frame.
📂 **resultados/** — reportes estadísticos, métricas globales y visualizaciones generadas durante el análisis.

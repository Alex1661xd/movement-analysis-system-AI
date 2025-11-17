"""
🎯 APLICACIÓN DE CLASIFICACIÓN DE MOVIMIENTO HUMANO
Sistema de Análisis de Actividades basado en MediaPipe y Machine Learning

Integrantes:
- Mariana De La Cruz - A00399618
- Valentina Gómez - A00398790
- Alexis Delgado - A00399176
- Juan Camilo Amorocho - A00399789

Entrega 3 - Proyecto Final APO 3
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Clasificador de Movimiento Humano",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #F8F9FA;
        border-left: 5px solid #2E86AB;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1976D2;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000000;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #F57C00;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000000;
    }
    .info-box h4, .warning-box h4 {
        color: #000000;
        font-weight: bold;
    }
    .info-box p, .warning-box p {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_resource
def cargar_modelos():
    """Carga todos los modelos y transformadores entrenados"""
    try:
        modelos_path = Path("resultados")
        
        modelos = {
            'random_forest': joblib.load(modelos_path / "random_forest_model.pkl"),
            'svm': joblib.load(modelos_path / "svm_model.pkl"),
            'xgboost': joblib.load(modelos_path / "xgboost_model.pkl"),
            'pca': joblib.load(modelos_path / "pca_model.pkl"),
            'scaler': joblib.load(modelos_path / "scaler_minmax.pkl"),
            'label_encoder': joblib.load(modelos_path / "label_encoder.pkl")
        }
        return modelos
    except Exception as e:
        st.error(f"❌ Error al cargar modelos: {e}")
        st.info("📌 Asegúrate de tener los archivos .pkl en la carpeta 'resultados/'")
        return None

def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos (a, b, c)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angulo = np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0)))
    return angulo

def extraer_metricas_video(video_path):
    """
    Extrae métricas biomecánicas de un video usando MediaPipe Pose
    Retorna un diccionario con las 12 características originales
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Variables de video
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion = total_frames / fps if fps > 0 else 0
    
    # Variables acumulativas
    total_brillo = 0
    total_mov = 0
    total_vel = 0
    total_aceleracion = 0
    total_ang_rod = 0
    total_ang_cad = 0
    total_ang_tob = 0
    total_incl = 0
    total_dist_hombros_caderas = 0
    total_simetria = 0
    
    frame_count = 0
    prev_gray = None
    prev_hip = None
    prev_hip_speed = 0
    frames_con_pose = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a escala de grises para brillo y movimiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total_brillo += np.mean(gray)
        
        # Movimiento entre frames
        if prev_gray is not None:
            total_mov += np.mean(cv2.absdiff(prev_gray, gray))
        prev_gray = gray
        
        # Procesar con MediaPipe
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            frames_con_pose += 1
            lm = results.pose_landmarks.landmark
            
            # Extraer puntos clave
            l_shoulder = np.array([lm[11].x, lm[11].y])
            r_shoulder = np.array([lm[12].x, lm[12].y])
            l_hip = np.array([lm[23].x, lm[23].y])
            r_hip = np.array([lm[24].x, lm[24].y])
            l_knee = np.array([lm[25].x, lm[25].y])
            l_ankle = np.array([lm[27].x, lm[27].y])
            l_foot = np.array([lm[31].x, lm[31].y])
            r_foot = np.array([lm[32].x, lm[32].y])
            
            # Métricas biomecánicas
            total_incl += abs(l_shoulder[1] - r_shoulder[1])
            total_ang_rod += calcular_angulo(l_hip, l_knee, l_ankle)
            total_ang_cad += calcular_angulo(l_shoulder, l_hip, l_knee)
            total_ang_tob += calcular_angulo(l_knee, l_ankle, l_foot)
            
            hombros_mid = (l_shoulder + r_shoulder) / 2
            caderas_mid = (l_hip + r_hip) / 2
            total_dist_hombros_caderas += np.linalg.norm(hombros_mid - caderas_mid)
            
            # Velocidad y aceleración de cadera
            hip_center = np.mean([l_hip, r_hip], axis=0)
            if prev_hip is not None:
                velocidad_actual = np.linalg.norm(hip_center - prev_hip)
                total_vel += velocidad_actual
                total_aceleracion += abs(velocidad_actual - prev_hip_speed)
                prev_hip_speed = velocidad_actual
            prev_hip = hip_center
            
            # Simetría corporal
            diff_izq_der = np.linalg.norm(l_hip - r_hip) + np.linalg.norm(l_shoulder - r_shoulder)
            total_simetria += diff_izq_der
        
        frame_count += 1
    
    cap.release()
    pose.close()
    
    # Calcular promedios
    if frame_count == 0 or frames_con_pose == 0:
        return None
    
    metricas = {
        'frames': total_frames,
        'duracion_seg': duracion,
        'brillo_promedio': total_brillo / frame_count,
        'movimiento_promedio': total_mov / max(frame_count - 1, 1),
        'velocidad_promedio': total_vel / frames_con_pose,
        'aceleracion_promedio': total_aceleracion / frames_con_pose,
        'angulo_rodilla_promedio': total_ang_rod / frames_con_pose,
        'angulo_cadera_promedio': total_ang_cad / frames_con_pose,
        'angulo_tobillo_promedio': total_ang_tob / frames_con_pose,
        'inclinacion_promedio': total_incl / frames_con_pose,
        'dist_hombros_caderas': total_dist_hombros_caderas / frames_con_pose,
        'simetria_promedio': total_simetria / frames_con_pose
    }
    
    return metricas, frames_con_pose, frame_count

def clasificar_video(metricas, modelos):
    """
    Clasifica un video usando los modelos entrenados
    Retorna predicciones de los 3 modelos y probabilidades
    """
    # Crear DataFrame con las métricas
    df_input = pd.DataFrame([metricas])
    
    # Normalizar con el scaler entrenado
    df_normalizado = modelos['scaler'].transform(df_input)
    
    # Aplicar PCA
    df_pca = modelos['pca'].transform(df_normalizado)
    
    # Hacer predicciones con los 3 modelos
    predicciones = {}
    
    for nombre, key in [('Random Forest', 'random_forest'), 
                        ('SVM', 'svm'), 
                        ('XGBoost', 'xgboost')]:
        modelo = modelos[key]
        pred_num = modelo.predict(df_pca)[0]
        pred_label = modelos['label_encoder'].inverse_transform([pred_num])[0]
        
        # Obtener probabilidades si el modelo lo soporta
        if hasattr(modelo, 'predict_proba'):
            try:
                proba = modelo.predict_proba(df_pca)[0]
                probabilidades = {
                    modelos['label_encoder'].inverse_transform([i])[0]: prob 
                    for i, prob in enumerate(proba)
                }
            except AttributeError:
                # SVM sin probability=True
                probabilidades = None
        else:
            probabilidades = None
        
        predicciones[nombre] = {
            'clase': pred_label,
            'probabilidades': probabilidades
        }
    
    return predicciones, df_pca

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Encabezado
    st.markdown('<div class="main-header"> Clasificador de Movimiento Humano</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema de Análisis de Actividades con IA - Entrega 3</div>', 
                unsafe_allow_html=True)
    
    # Cargar modelos
    with st.spinner('🔄 Cargando modelos de Machine Learning...'):
        modelos = cargar_modelos()
    
    if modelos is None:
        st.stop()
    
    st.success('✅ Modelos cargados correctamente')
    
    # Sidebar con información
    with st.sidebar:
        st.header("📋 Información del Sistema")
        
        st.markdown("""
        ### 🎯 Actividades que puede clasificar:
        - **Adelante**: Caminar hacia la cámara
        - **Atrás**: Caminar alejándose
        - **Sentado**: Posición sentada
        - **Cadera al frente**: Flexión de cadera frontal
        - **Caderas**: Rotación de caderas
        - **Lado**: Movimiento lateral
        - **Sentadilla**: Sentadilla profunda
        - **Tijeras**: Movimiento de tijeras
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📊 Modelos Utilizados:
        - **Random Forest** (94.44% accuracy)
        - **SVM RBF** (94.44% accuracy)
        - **XGBoost** (94.44% accuracy)
        
        *Entrenados con reducción dimensional PCA (6 componentes)*
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 👥 Equipo de Desarrollo:
        - Mariana De La Cruz
        - Valentina Gómez
        - Alexis Delgado
        - Juan Camilo Amorocho
        """)
    
    # Área principal
    st.markdown("---")
    
    # Instrucciones
    with st.expander("Cómo usar esta aplicación", expanded=True):
        st.markdown("""
        1. **Sube un video** usando el botón de abajo
        2. El sistema extraerá automáticamente las **características biomecánicas** usando MediaPipe
        3. Los **3 modelos entrenados** clasificarán el movimiento
        4. Verás los **resultados detallados** con gráficas y métricas
        
        **Recomendaciones:**
        - Videos de 2-5 segundos funcionan mejor
        - Asegúrate que la persona sea visible de cuerpo completo
        - Buena iluminación mejora la detección
        """)
    
    st.markdown("---")
    
    # Upload de video
    st.subheader("Subir Video para Clasificar")
    
    uploaded_file = st.file_uploader(
        "Arrastra tu video aquí o haz clic para seleccionar",
        type=['mp4', 'avi', 'mov'],
        help="Formatos soportados: MP4, AVI, MOV"
    )
    
    if uploaded_file is not None:
        # Guardar video temporalmente
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Mostrar video
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(video_path)
        
        with col2:
            st.info(f"""
            **Información del archivo:**
            - **Nombre:** {uploaded_file.name}
            - **Tamaño:** {uploaded_file.size / 1024:.1f} KB
            """)
        
        st.markdown("---")
        
        # Botón de procesar con estado de carga
        analizar_clicked = st.button(" Analizar Video", type="primary", use_container_width=True)
        
        if analizar_clicked:
            # Deshabilitar interacción durante el procesamiento
            with st.spinner('⏳ Analizando video... Por favor espera'):
                
                # Barra de progreso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Paso 1: Extraer métricas
                status_text.text("Paso 1/3: Extrayendo landmarks con MediaPipe...")
                progress_bar.progress(33)
                
                resultado = extraer_metricas_video(video_path)
                
                if resultado is None:
                    st.error("❌ No se pudo procesar el video. Asegúrate de que haya una persona visible.")
                    os.unlink(video_path)
                    st.stop()
                
                metricas, frames_con_pose, total_frames = resultado
                
                # Paso 2: Clasificar
                status_text.text("Paso 2/3: Clasificando con modelos de IA...")
                progress_bar.progress(66)
                
                predicciones, componentes_pca = clasificar_video(metricas, modelos)
                
                # Paso 3: Mostrar resultados
                status_text.text("Paso 3/3: Generando resultados...")
                progress_bar.progress(100)
                
                # Limpiar barra de progreso
                progress_bar.empty()
                status_text.empty()
            
            # ================================================================
            # MOSTRAR RESULTADOS
            # ================================================================
            
            st.markdown("---")
            st.header(" Resultados de Clasificación")
            
            # Determinar predicción por consenso
            votos = [pred['clase'] for pred in predicciones.values()]
            from collections import Counter
            prediccion_final = Counter(votos).most_common(1)[0][0]
            
            # Predicción principal
            st.markdown(f"""
            <div class="prediction-box">
                 ACTIVIDAD DETECTADA: {prediccion_final.upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Resultados por modelo
            st.subheader(" Predicciones por Modelo")
            
            cols = st.columns(3)
            
            for idx, (nombre, resultado) in enumerate(predicciones.items()):
                with cols[idx]:
                    color = "🟢" if resultado['clase'] == prediccion_final else "🟡"
                    st.markdown(f"### {color} {nombre}")
                    st.markdown(f"**Predicción:** `{resultado['clase']}`")
                    
                    if resultado['probabilidades'] is not None:
                        st.markdown("**Confianza:**")
                        probs_sorted = sorted(
                            resultado['probabilidades'].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        for clase, prob in probs_sorted[:3]:
                            # Convertir a float de Python para evitar error con numpy.float32
                            prob_float = float(prob)
                            st.progress(prob_float, text=f"{clase}: {prob_float*100:.1f}%")
                    else:
                        st.info("ℹ️ Este modelo no proporciona probabilidades")
            
            st.markdown("---")
            
            # Métricas biomecánicas
            st.subheader("Métricas Biomecánicas Extraídas")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Duración", f"{metricas['duracion_seg']:.2f} s")
                st.metric("Frames totales", metricas['frames'])
                st.metric("Poses detectadas", frames_con_pose)
            
            with col2:
                st.metric("Velocidad cadera", f"{metricas['velocidad_promedio']:.4f}")
                st.metric("Aceleración", f"{metricas['aceleracion_promedio']:.4f}")
                st.metric("Movimiento", f"{metricas['movimiento_promedio']:.2f}")
            
            with col3:
                st.metric("Ángulo rodilla", f"{metricas['angulo_rodilla_promedio']:.1f}°")
                st.metric("Ángulo cadera", f"{metricas['angulo_cadera_promedio']:.1f}°")
                st.metric("Ángulo tobillo", f"{metricas['angulo_tobillo_promedio']:.1f}°")
            
            with col4:
                st.metric("Inclinación", f"{metricas['inclinacion_promedio']:.4f}")
                st.metric("Dist. hombros-caderas", f"{metricas['dist_hombros_caderas']:.4f}")
                st.metric("Simetría corporal", f"{metricas['simetria_promedio']:.4f}")
            
            # Gráfica de componentes PCA
            st.markdown("---")
            st.subheader("Análisis de Componentes Principales (PCA)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                componentes = [f'PC{i+1}' for i in range(len(componentes_pca[0]))]
                valores = componentes_pca[0]
                
                colors = ['#2E86AB' if v >= 0 else '#E63946' for v in valores]
                ax.bar(componentes, valores, color=colors, edgecolor='black', linewidth=1.5)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel('Componente Principal', fontweight='bold', fontsize=12)
                ax.set_ylabel('Valor', fontweight='bold', fontsize=12)
                ax.set_title('Valores de los Componentes Principales del Video', 
                            fontweight='bold', fontsize=14)
                ax.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig)
            
            with col2:
                st.markdown("""
                ### Interpretación
                
                Los **6 componentes principales** representan combinaciones de las 12 características originales.
                
                **Significado:**
                - **PC1**: Ángulos corporales principales
                - **PC2**: Duración y brillo
                - **PC3**: Distancia hombros-caderas
                - **PC4**: Movimiento general
                - **PC5**: Brillo y tiempo
                - **PC6**: Velocidad y aceleración
                """)
            
            # Información adicional
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>Calidad de Detección</h4>
                <p>Se detectaron landmarks en <strong>{:.1f}%</strong> de los frames ({}/{} frames).</p>
                <p>Esto indica una <strong>buena calidad de detección</strong> para la clasificación.</p>
                </div>
                """.format(
                    (frames_con_pose / total_frames) * 100,
                    frames_con_pose,
                    total_frames
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="warning-box">
                <h4>⚠️ Nota Importante</h4>
                <p>Los modelos fueron entrenados con un dataset limitado (86 videos).</p>
                <p>Para mejores resultados en producción, se recomienda entrenar con más datos.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Botón para descargar resultados
            st.markdown("---")
            
            resultados_dict = {
                'Predicción Final': prediccion_final,
                'Random Forest': predicciones['Random Forest']['clase'],
                'SVM': predicciones['SVM']['clase'],
                'XGBoost': predicciones['XGBoost']['clase'],
                **metricas
            }
            
            df_resultados = pd.DataFrame([resultados_dict])
            csv = df_resultados.to_csv(index=False)
            
            st.download_button(
                label="📥 Descargar Resultados (CSV)",
                data=csv,
                file_name=f"clasificacion_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Limpiar archivo temporal
        try:
            os.unlink(video_path)
        except:
            pass
    
    else:
        st.info("👆 Sube un video para comenzar el análisis")

# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    main()


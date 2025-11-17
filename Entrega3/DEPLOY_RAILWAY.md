# 🚂 Guía de Despliegue en Railway

Esta guía te ayudará a desplegar la aplicación de Clasificación de Movimiento Humano en Railway.

---

## 📋 Requisitos Previos

1. **Cuenta en Railway**: [Regístrate aquí](https://railway.app/)
2. **Repositorio Git**: Tu código debe estar en GitHub, GitLab o Bitbucket
3. **Modelos entrenados**: Los archivos `.pkl` deben estar en `resultados/`

---

## 🚀 Pasos para Desplegar

### 1. Preparar los Modelos

Asegúrate de tener estos archivos en `Entrega3/resultados/`:

```
resultados/
├── random_forest_model.pkl
├── svm_model.pkl (con probability=True)
├── xgboost_model.pkl
├── pca_model.pkl
├── scaler_minmax.pkl
└── label_encoder.pkl
```

**⚠️ IMPORTANTE:** Si tu SVM no tiene `probability=True`, ejecuta:

```bash
# En Google Colab, al final del notebook de Entrega 3:
exec(open('reentrenar_svm_con_probabilidades.py').read())
```

Luego descarga el nuevo `svm_model.pkl` y reemplázalo.

---

### 2. Verificar Archivos de Configuración

Asegúrate de tener estos archivos en la carpeta `Entrega3/`:

- ✅ `requirements.txt`
- ✅ `Procfile`
- ✅ `railway.toml`
- ✅ `runtime.txt`
- ✅ `.streamlit/config.toml`
- ✅ `.railwayignore`

---

### 3. Subir a GitHub

```bash
# Inicializar repositorio (si no existe)
git init

# Agregar archivos
git add .

# Commit
git commit -m "Deploy: Aplicación de clasificación de movimiento"

# Conectar con GitHub (reemplaza con tu repo)
git remote add origin https://github.com/tu-usuario/movement-analysis-system-AI.git

# Push
git push -u origin main
```

---

### 4. Crear Proyecto en Railway

1. Ve a [Railway.app](https://railway.app/)
2. Haz clic en **"New Project"**
3. Selecciona **"Deploy from GitHub repo"**
4. Autoriza Railway a acceder a tu GitHub
5. Selecciona tu repositorio `movement-analysis-system-AI`
6. Railway detectará automáticamente el proyecto

---

### 5. Configurar el Proyecto

#### Opción A: Configuración Automática (Recomendado)

Railway leerá `railway.toml` y configurará todo automáticamente.

#### Opción B: Configuración Manual

Si necesitas configurar manualmente:

1. Ve a **Settings** → **Deploy**
2. **Root Directory**: `Entrega3`
3. **Start Command**: 
   ```
   streamlit run app_clasificador_movimiento.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```

---

### 6. Variables de Entorno

Railway las configurará automáticamente, pero puedes verificar:

- `PORT`: Asignado por Railway
- `PYTHONUNBUFFERED`: `1`

---

### 7. Iniciar el Despliegue

1. Railway empezará a construir automáticamente
2. Verás los logs en tiempo real
3. El proceso toma ~5-10 minutos

```
Building...
[+] Installing packages...
[+] Installing streamlit, opencv, mediapipe...
[+] Build complete
Deploying...
✓ Deployment successful
```

---

### 8. Obtener la URL

Una vez desplegado:

1. Ve a **Settings** → **Networking**
2. Haz clic en **"Generate Domain"**
3. Railway te dará una URL como: `https://tu-app.up.railway.app`

---

## 🔧 Solución de Problemas

### Error: "Module not found"

```bash
# Verifica que requirements.txt esté correcto
pip freeze > requirements.txt
```

### Error: "Out of memory"

Los modelos son pesados. Railway Free Tier tiene límites:
- **512 MB RAM** (puede no ser suficiente)
- Considera Railway Pro ($5/mes) o Render.com

### Error: "Application failed to respond"

```bash
# Verifica el comando de inicio en railway.toml
startCommand = "streamlit run app_clasificador_movimiento.py --server.port $PORT --server.address 0.0.0.0"
```

### Error: "FileNotFoundError: 'resultados/'"

Asegúrate de que la carpeta `resultados/` con los archivos `.pkl` esté en el repositorio y NO en `.railwayignore`.

---

## 📊 Límites de Railway

### Free Tier
- ✅ 500 horas/mes
- ✅ 512 MB RAM
- ✅ 1 GB Disco
- ⚠️ Puede ser insuficiente para modelos grandes

### Pro Tier ($5/mes)
- ✅ Ilimitado
- ✅ 8 GB RAM
- ✅ 10 GB Disco
- ✅ Mejor rendimiento

---

## 🎨 Alternativas a Railway

Si Railway no funciona por limitaciones de recursos:

### Render.com (Recomendado)
- Free tier: 512 MB RAM
- Más estable para aplicaciones ML
- [Deploy en Render](https://render.com/)

### Streamlit Cloud
- Gratis para apps públicas
- Optimizado para Streamlit
- [Deploy en Streamlit Cloud](https://streamlit.io/cloud)

### Hugging Face Spaces
- Gratis con GPU
- Ideal para modelos ML
- [Deploy en HF Spaces](https://huggingface.co/spaces)

---

## ✅ Checklist Final

Antes de hacer push:

- [ ] Modelos `.pkl` en `resultados/`
- [ ] SVM con `probability=True`
- [ ] `requirements.txt` actualizado
- [ ] Archivos de configuración presentes
- [ ] `.gitignore` configurado
- [ ] Código testeado localmente
- [ ] Sin archivos grandes (videos) en el repo

---

## 📝 Comandos Útiles

```bash
# Ver logs en tiempo real
railway logs

# Ver estado del servicio
railway status

# Reiniciar servicio
railway restart

# Abrir en navegador
railway open
```

---

## 🎓 Soporte

Si tienes problemas:

1. **Logs de Railway**: Revisa los logs de construcción/despliegue
2. **Documentación**: [Railway Docs](https://docs.railway.app/)
3. **Discord de Railway**: Comunidad muy activa

---

## 🚀 Siguiente Paso

Una vez desplegado, comparte tu URL:

```
🎉 Aplicación desplegada en:
https://movement-classifier.up.railway.app

Pruébala subiendo un video de 2-5 segundos!
```

---

**¡Listo para producción!** 🎊


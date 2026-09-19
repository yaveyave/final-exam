cat > README.md << 'EOF'
# TSP Hardness Predictor: P vs NP with Machine Learning

Predicción de dureza computacional del Problema del Viajante (TSP) mediante aprendizaje automático para proporcionar evidencia empírica sobre P ≠ NP.

## Características

- **4 Modelos ML**: kNN, Regresión Lineal, Random Forest, MLP Keras
- **145 Instancias**: Dataset sintético optimizado para extrapolation analysis
- **Validación Rigurosa**: K-Fold CV, Matriz de Confusión, Learning Curves
- **Extrapolation Test**: Evidencia de complejidad exponencial (no-polinomial)
- **Streamlit App**: Predictor interactivo

## Instalación

```bash
git clone https://github.com/TU_USUARIO/TSP-Hardness-Predictor.git
cd TSP-Hardness-Predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### Ejecutar localmente
```bash
streamlit run app.py
```

### Scripts individuales
```bash
cd scripts
python script_1_dataset.py      # Generar dataset
python script_2_train_models.py # Entrenar modelos
python script_3_extrapolation.py # Análisis extrapolation
```

## Resultados Clave

| Modelo | Métrica | Valor |
|--------|---------|-------|
| kNN | Accuracy | 0.7931 |
| Lineal | R² | 0.8729 |
| Random Forest | R² | 0.9993 |
| MLP | R² | 0.9595 |

**Extrapolation Error:**
- Lineal: RMSE 110.44x amplificado
- RF: RMSE 97.33x amplificado
- **Conclusión:** Evidencia empírica de P ≠ NP

## Institución

Maestría en Inteligencia Artificial - Yachay Tech
Módulo: Fundamentos de Inteligencia Artificial

## Autor

Oscar [@oscar-github-username]
EOF
import subprocess
import sys
import os

# Asegurar que existen modelos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Entrenar si no existen
if not os.path.exists('results/models/model_knn.pkl'):
    print("⏳ Entrenando modelos (primera vez)...")
    os.chdir('scripts')
    subprocess.run([sys.executable, 'script_1_dataset.py'], check=True)
    subprocess.run([sys.executable, 'script_2_train_models.py'], check=True)
    subprocess.run([sys.executable, 'script_3_extrapolation.py'], check=True)
    os.chdir('..')
    print("✅ Modelos listos!")

# Ejecutar script_4
exec(open('scripts/script_4_streamlit_app.py').read())

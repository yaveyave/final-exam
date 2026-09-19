import subprocess
import sys
import os
import runpy

# Obtener ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')

sys.path.insert(0, SCRIPTS_DIR)

# Entrenar modelos si no existen
if not os.path.exists(os.path.join(MODELS_DIR, 'model_knn.pkl')):
    print("⏳ Entrenando modelos (primera vez)...")
    script_1 = os.path.join(SCRIPTS_DIR, 'script_1_dataset.py')
    script_2 = os.path.join(SCRIPTS_DIR, 'script_2_train_models.py')
    script_3 = os.path.join(SCRIPTS_DIR, 'script_3_extrapolation.py')
    
    subprocess.run([sys.executable, script_1], cwd=BASE_DIR, check=True)
    subprocess.run([sys.executable, script_2], cwd=BASE_DIR, check=True)
    subprocess.run([sys.executable, script_3], cwd=BASE_DIR, check=True)
    print("✅ Modelos listos!")

# Ejecutar app
app_script = os.path.join(SCRIPTS_DIR, 'script_4_streamlit_app.py')
runpy.run_path(app_script, run_name='__main__')

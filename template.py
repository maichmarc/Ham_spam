import os
from pathlib import Path



list_of_files = [
    '__init__.py',
    'components/__init__.py',
    'components/data_ingestion.py',
    'components/data_transformation.py',
    'components/model_trainer.py',
    'notebooks/'
    'static/css/style.css',
    'templates/home.html',
    'templates/index.html',
    'pipeline/__init__.py',
    'pipeline/training_pipeline.py',
    'pipeline/prediction_pipeline.py',
    'exception.py',
    'logger.py',
    'utils.py',
    'app.py',
    'requirements.txt',   
    'setup.py'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
    else:
        print(f'file is already present at {filepath}')

# File: components/version_control.py
# Version: v1.16

import os
from datetime import datetime

# Обновленный словарь с информацией о версиях и последнем изменении файлов
COMPONENTS_INFO = {
    'dataset_builder.py': {'version': 'v1.0', 'last_modified': '2025-07-11'},
    'components/main.py': {'version': 'v1.4', 'last_modified': '2025-07-11'},
    'components/indicators_metrics_builder.py': {'version': 'v0.26', 'last_modified': '2025-07-12'},
    'components/data_validation.py': {'version': 'v0.11', 'last_modified': '2025-07-12'},
    'components/data_transformation.py': {'version': 'v2.1', 'last_modified': '2025-07-11'},
    'components/debug.py': {'version': 'v0.3', 'last_modified': '2025-07-11'},
    'connectors/main.py': {'version': 'v1.2', 'last_modified': '2025-07-11'},
    'connectors/mt5_api.py': {'version': 'v1.3', 'last_modified': '2025-07-11'},
}

def check_version(file_path):
    """Проверяет версию файла и дату последнего изменения."""
    # Определяем относительное имя: для файлов внутри components/ или connectors/
    dirname = os.path.basename(os.path.dirname(file_path))
    basename = os.path.basename(file_path)
    if dirname in ('components', 'connectors'):
        file_key = f"{dirname}/{basename}"
    else:
        file_key = basename

    if file_key in COMPONENTS_INFO:
        component_info = COMPONENTS_INFO[file_key]
        version = component_info['version']
        last_modified = component_info['last_modified']
        actual_modified = datetime.fromtimestamp(
            os.path.getmtime(file_path)
        ).strftime('%Y-%m-%d')
        if actual_modified > last_modified:
            print(f"[ERROR] WARNING: {file_key} was modified on {actual_modified} "
                  f"as version {version} with last known update on {last_modified}")
        else:
            print(f"[INFO] {file_key} is up-to-date with version {version}")
    else:
        print(f"[ERROR] Version information for {file_key} not found.")
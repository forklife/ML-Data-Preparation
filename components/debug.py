from tqdm import tqdm
DEBUG, INFO, WARNING, ERROR = 0, 1, 2, 3
levels = {DEBUG: 'DEBUG', INFO: 'INFO', WARNING: 'WARN', ERROR: 'ERROR'}

# File: components/debug.py
# Version: v0.3

from components.version_control import check_version

# Определение file_path для текущего файла
file_path = __file__

# Проверка версии при запуске
check_version(file_path)

# Уровни логирования
DEBUG = 1
INFO = 2
ERROR = 3

# Текущий уровень логирования (можно менять)
CURRENT_LEVEL = DEBUG

# Цветовые коды для вывода сообщений
COLORS = {
    DEBUG: '\033[94m',  # Синий цвет для отладочных сообщений
    INFO: '\033[92m',   # Зеленый цвет для информации
    ERROR: '\033[91m',  # Красный цвет для ошибок
    'ENDC': '\033[0m',  # Сброс цвета
}

def log(message, level=INFO):
    color = COLORS.get(level, COLORS['ENDC'])
    formatted = f"{color}[{levels.get(level, level)}] {message}{COLORS['ENDC']}"
    try:
        tqdm.write(formatted)
    except Exception:
        print(formatted)

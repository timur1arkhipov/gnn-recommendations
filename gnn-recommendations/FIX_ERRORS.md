# 🔧 Исправление ошибок

## Проблема 1: `'Trainer' object has no attribute 'epochs'`

✅ **ИСПРАВЛЕНО!** Обновите файл `src/training/trainer.py` (уже сделано).

## Проблема 2: `No module named 'matplotlib'`

Нужно установить недостающие библиотеки.

### Решение:

**Вариант 1 (простой):**
```bash
pip install matplotlib seaborn pandas scipy
```

**Вариант 2 (через requirements.txt):**
```bash
pip install -r requirements.txt
```

**Вариант 3 (через bat-файл):**
```bash
install_dependencies.bat
```

## Проверка

После установки проверьте:

```bash
python scripts/check_gpu.py
```

Если всё работает, запускайте эксперименты:

```bash
python run_all.py --quick
```

## Что было исправлено

1. ✅ Порядок инициализации в `Trainer.__init__()` - `self.epochs` теперь определяется ДО использования
2. ✅ Создан `install_dependencies.bat` для установки библиотек
3. ✅ Все импорты исправлены

## Если всё ещё есть ошибки

### Ошибка импорта

Убедитесь, что запускаете из правильной директории:

```bash
cd gnn-recommendations
python run_all.py
```

### Ошибка CUDA

Если нет GPU:
```bash
pip install torch torchvision torchaudio
```

Если есть GPU (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Быстрая проверка

```bash
# 1. Установите зависимости
pip install matplotlib seaborn pandas scipy

# 2. Проверьте GPU
python scripts/check_gpu.py

# 3. Запустите быстрый тест
python run_all.py --quick
```

Готово! 🎉


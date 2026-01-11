@echo off
echo.
echo ================================================================================
echo УСТАНОВКА ЗАВИСИМОСТЕЙ
echo ================================================================================
echo.

cd /d "%~dp0"

echo Установка основных библиотек...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Установка дополнительных библиотек...
pip install numpy pandas scipy matplotlib seaborn pyyaml

echo.
echo ================================================================================
echo ГОТОВО!
echo ================================================================================
echo.
echo Проверьте установку:
python scripts\check_gpu.py
echo.

pause


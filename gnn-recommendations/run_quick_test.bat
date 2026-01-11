@echo off
REM ============================================================================
REM Быстрый тест - запуск нескольких моделей с 2 seeds на MovieLens
REM Для проверки, что все работает (займет ~30 минут)
REM ============================================================================

echo.
echo ================================================================================
echo БЫСТРЫЙ ТЕСТ (2 seeds, только MovieLens)
echo ================================================================================
echo.

cd /d "%~dp0"
set PYTHONPATH=%CD%

echo [1/3] Проверка GPU...
python scripts\check_gpu.py

echo.
echo [2/3] Запуск экспериментов (2 seeds)...
echo.

python scripts\run_multiple_seeds.py ^
    --models bpr_mf lightgcn groupshuffle_gnn ^
    --datasets movie_lens ^
    --seeds 42 43 ^
    --baseline_model lightgcn

echo.
echo [3/3] Создание графиков...
echo.

python scripts\analyze_and_plot.py ^
    --results_dir results\multiple_seeds ^
    --output_dir results\figures ^
    --baseline_model lightgcn

echo.
echo ================================================================================
echo ТЕСТ ЗАВЕРШЕН!
echo ================================================================================
echo Результаты в: results\figures\
echo.

pause


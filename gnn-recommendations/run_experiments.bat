@echo off
REM ============================================================================
REM Скрипт для запуска всех экспериментов для магистерской диссертации
REM ============================================================================

echo.
echo ================================================================================
echo ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ ДЛЯ МАГИСТЕРСКОЙ ДИССЕРТАЦИИ
echo ================================================================================
echo.

REM Переходим в директорию проекта
cd /d "%~dp0"

REM Устанавливаем PYTHONPATH
set PYTHONPATH=%CD%

echo [1/6] Проверка GPU...
python scripts\check_gpu.py
if errorlevel 1 (
    echo ОШИБКА: Проблемы с GPU!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [2/6] ОСНОВНЫЕ ЭКСПЕРИМЕНТЫ (5 seeds)
echo ================================================================================
echo Это займет около 2-3 часов для MovieLens и Book Crossing
echo.

python scripts\run_multiple_seeds.py ^
    --models bpr_mf lightgcn gcnii dgr layergcn groupshuffle_gnn ^
    --datasets movie_lens book_crossing ^
    --seeds 42 43 44 45 46 ^
    --baseline_model layergcn

if errorlevel 1 (
    echo ОШИБКА при запуске основных экспериментов!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [3/6] DEPTH ANALYSIS для GroupShuffleGNN
echo ================================================================================
echo.

python scripts\run_depth_analysis.py ^
    --model groupshuffle_gnn ^
    --dataset movie_lens ^
    --layers 2 4 8 16

if errorlevel 1 (
    echo ОШИБКА при depth analysis!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [4/6] ABLATION STUDIES для GroupShuffleGNN
echo ================================================================================
echo.

python scripts\run_ablations.py --dataset movie_lens

if errorlevel 1 (
    echo ОШИБКА при ablation studies!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [5/6] АНАЛИЗ И ВИЗУАЛИЗАЦИЯ
echo ================================================================================
echo.

python scripts\analyze_and_plot.py ^
    --results_dir results\multiple_seeds ^
    --output_dir results\figures ^
    --baseline_model layergcn ^
    --metrics recall@10 ndcg@10 precision@10 coverage

if errorlevel 1 (
    echo ОШИБКА при создании графиков!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [6/6] CASE STUDY
echo ================================================================================
echo.

python scripts\generate_case_study.py ^
    --dataset movie_lens ^
    --models bpr_mf lightgcn layergcn groupshuffle_gnn ^
    --n_users 10 ^
    --k 10

if errorlevel 1 (
    echo ОШИБКА при генерации case study!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ УСПЕШНО!
echo ================================================================================
echo.
echo Результаты сохранены в:
echo   - results\multiple_seeds\     (основные результаты)
echo   - results\figures\            (графики и таблицы)
echo   - experiments\depth_analysis\ (анализ глубины)
echo   - experiments\ablations\      (ablation studies)
echo   - results\case_study\         (примеры рекомендаций)
echo.
echo Откройте EXPERIMENTS_GUIDE.md для подробной информации
echo.

pause


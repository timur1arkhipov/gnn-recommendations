"""
Главный запускатор всех экспериментов для магистерской диссертации.

Просто запустите:
    python run_all.py

Или для быстрого теста:
    python run_all.py --quick
"""

import sys
import subprocess
from pathlib import Path
import argparse
import time

# Корневая директория проекта
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(description, command, cwd=None):
    """Запускает команду и выводит результат."""
    print("\n" + "="*80)
    print(f"▶ {description}")
    print("="*80)
    print(f"Команда: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd or PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"\n✅ {description} - УСПЕШНО")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - ОШИБКА")
        print(f"Код возврата: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ОШИБКА: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Запуск всех экспериментов для магистерской диссертации"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый тест (2 seeds, только MovieLens, 3 модели)"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Пропустить проверку GPU"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 ЗАПУСК ЭКСПЕРИМЕНТОВ ДЛЯ МАГИСТЕРСКОЙ ДИССЕРТАЦИИ")
    print("="*80)
    
    if args.quick:
        print("Режим: БЫСТРЫЙ ТЕСТ (2 seeds, MovieLens, 3 модели)")
        print("Ожидаемое время: ~30 минут")
    else:
        print("Режим: ПОЛНЫЙ ЦИКЛ (5 seeds, 2 датасета, 7 моделей)")
        print("Ожидаемое время: ~15-20 часов")
    
    print("="*80)
    
    start_time = time.time()
    
    # Шаг 1: Проверка GPU
    if not args.skip_check:
        success = run_command(
            "Проверка GPU",
            [sys.executable, "scripts/check_gpu.py"]
        )
        if not success:
            print("\n⚠️  Проблемы с GPU, но продолжаем...")
    
    # Конфигурация для быстрого теста или полного цикла
    if args.quick:
        models = ["bpr_mf", "lightgcn", "groupshuffle_gnn"]
        datasets = ["movie_lens"]
        seeds = ["42", "43"]
        baseline = "lightgcn"
    else:
        models = ["bpr_mf", "lightgcn", "gcnii", "dgr", "layergcn", "groupshuffle_gnn"]
        datasets = ["movie_lens", "book_crossing"]
        seeds = ["42", "43", "44", "45", "46"]
        baseline = "layergcn"
    
    # Шаг 2: Основные эксперименты (multiple seeds)
    success = run_command(
        f"Основные эксперименты ({len(seeds)} seeds)",
        [
            sys.executable, "scripts/run_multiple_seeds.py",
            "--models"] + models + [
            "--datasets"] + datasets + [
            "--seeds"] + seeds + [
            "--baseline_model", baseline
        ]
    )
    
    if not success:
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА при основных экспериментах!")
        print("Проверьте логи выше для деталей.")
        return 1
    
    # Шаг 3: Depth Analysis (только для полного цикла)
    if not args.quick:
        success = run_command(
            "Depth Analysis для GroupShuffleGNN",
            [
                sys.executable, "scripts/run_depth_analysis.py",
                "--model", "groupshuffle_gnn",
                "--dataset", "movie_lens",
                "--layers", "2", "4", "8", "16"
            ]
        )
        
        if not success:
            print("\n⚠️  Ошибка в Depth Analysis, но продолжаем...")
    
    # Шаг 4: Ablation Studies (только для полного цикла)
    if not args.quick:
        success = run_command(
            "Ablation Studies для GroupShuffleGNN",
            [
                sys.executable, "scripts/run_ablations.py",
                "--dataset", "movie_lens"
            ]
        )
        
        if not success:
            print("\n⚠️  Ошибка в Ablation Studies, но продолжаем...")
    
    # Шаг 5: Анализ и визуализация
    success = run_command(
        "Создание графиков и таблиц",
        [
            sys.executable, "scripts/analyze_and_plot.py",
            "--results_dir", "results/multiple_seeds",
            "--output_dir", "results/figures",
            "--baseline_model", baseline,
            "--metrics", "recall@10", "ndcg@10", "precision@10", "coverage"
        ]
    )
    
    if not success:
        print("\n⚠️  Ошибка при создании графиков, но результаты сохранены")
    
    # Шаг 6: Case Study (только для полного цикла)
    if not args.quick:
        success = run_command(
            "Генерация Case Study",
            [
                sys.executable, "scripts/generate_case_study.py",
                "--dataset", "movie_lens",
                "--models", "bpr_mf", "lightgcn", "layergcn", "groupshuffle_gnn",
                "--n_users", "10",
                "--k", "10"
            ]
        )
        
        if not success:
            print("\n⚠️  Ошибка в Case Study, но основные результаты готовы")
    
    # Итоги
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print("\n" + "="*80)
    print("🎉 ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("="*80)
    print(f"Общее время: {hours}ч {minutes}мин")
    print("\nРезультаты сохранены в:")
    print("  📁 results/multiple_seeds/     - основные результаты (JSON)")
    print("  📊 results/figures/            - графики и LaTeX таблицы")
    
    if not args.quick:
        print("  📈 experiments/depth_analysis/ - анализ глубины")
        print("  🔬 experiments/ablations/      - ablation studies")
        print("  📝 results/case_study/         - примеры рекомендаций")
    
    print("\n📖 Откройте EXPERIMENTS_GUIDE.md для подробной информации")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


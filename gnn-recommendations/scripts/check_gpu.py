"""
Скрипт для проверки доступности и работоспособности GPU.

Использование:
    python scripts/check_gpu.py
"""

import torch
import sys
from pathlib import Path

# Добавляем путь к src
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))


def check_cuda_available():
    """Проверяет доступность CUDA."""
    print("=" * 60)
    print("ПРОВЕРКА CUDA")
    print("=" * 60)
    
    is_available = torch.cuda.is_available()
    print(f"\n✓ CUDA доступна: {is_available}")
    
    if not is_available:
        print("\n❌ CUDA недоступна!")
        print("\nВозможные причины:")
        print("1. PyTorch установлен без поддержки CUDA")
        print("2. Драйверы NVIDIA не установлены")
        print("3. GPU не поддерживает CUDA")
        print("\nУстановка PyTorch с CUDA:")
        print("  pip uninstall torch torchvision torchaudio")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    return True


def check_cuda_info():
    """Выводит информацию о CUDA."""
    print("\n" + "=" * 60)
    print("ИНФОРМАЦИЯ О CUDA")
    print("=" * 60)
    
    print(f"\n✓ Версия CUDA: {torch.version.cuda}")
    print(f"✓ Версия cuDNN: {torch.backends.cudnn.version()}")
    print(f"✓ Количество GPU: {torch.cuda.device_count()}")


def check_gpu_info():
    """Выводит информацию о GPU."""
    print("\n" + "=" * 60)
    print("ИНФОРМАЦИЯ О GPU")
    print("=" * 60)
    
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        print(f"\n🎮 GPU {i}:")
        print(f"  Название: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Общая память: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Текущее использование памяти
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  Память выделена: {memory_allocated:.2f} GB")
        print(f"  Память зарезервирована: {memory_reserved:.2f} GB")


def test_simple_operation():
    """Тестирует простые операции на GPU."""
    print("\n" + "=" * 60)
    print("ТЕСТ ПРОСТЫХ ОПЕРАЦИЙ")
    print("=" * 60)
    
    device = torch.device('cuda:0')
    print(f"\n✓ Используемое устройство: {device}")
    
    # Создаем тензоры на GPU
    print("\n1. Создание тензоров на GPU...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    print(f"   ✓ Тензор x: shape={x.shape}, device={x.device}")
    print(f"   ✓ Тензор y: shape={y.shape}, device={y.device}")
    
    # Матричное умножение
    print("\n2. Матричное умножение на GPU...")
    z = torch.mm(x, y)
    print(f"   ✓ Результат z: shape={z.shape}, device={z.device}")
    
    # Перенос на CPU
    print("\n3. Перенос результата на CPU...")
    z_cpu = z.cpu()
    print(f"   ✓ Результат на CPU: shape={z_cpu.shape}, device={z_cpu.device}")
    
    print("\n✅ Все операции выполнены успешно!")


def benchmark_speed():
    """Сравнивает скорость CPU vs GPU."""
    print("\n" + "=" * 60)
    print("БЕНЧМАРК: CPU vs GPU")
    print("=" * 60)
    
    import time
    
    size = 5000
    iterations = 10
    
    # CPU
    print(f"\n⏱️  CPU бенчмарк ({iterations} итераций, матрица {size}x{size})...")
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    start = time.time()
    for _ in range(iterations):
        z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"   Время CPU: {cpu_time:.2f} секунд")
    
    # GPU
    print(f"\n⚡ GPU бенчмарк ({iterations} итераций, матрица {size}x{size})...")
    x_gpu = torch.randn(size, size, device='cuda')
    y_gpu = torch.randn(size, size, device='cuda')
    
    # Прогрев GPU
    for _ in range(3):
        _ = torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        z_gpu = torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"   Время GPU: {gpu_time:.2f} секунд")
    
    # Сравнение
    speedup = cpu_time / gpu_time
    print(f"\n🚀 Ускорение: {speedup:.2f}x")
    
    if speedup > 5:
        print("   ✅ Отличное ускорение! GPU работает правильно.")
    elif speedup > 2:
        print("   ⚠️  Умеренное ускорение. Возможно, есть узкие места.")
    else:
        print("   ❌ Слабое ускорение. Проверьте настройки GPU.")


def test_model_training():
    """Тестирует обучение простой модели на GPU."""
    print("\n" + "=" * 60)
    print("ТЕСТ ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 60)
    
    import torch.nn as nn
    import torch.optim as optim
    
    device = torch.device('cuda')
    
    # Простая модель
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1000, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    print("\n1. Создание модели...")
    model = SimpleModel().to(device)
    print(f"   ✓ Модель на устройстве: {next(model.parameters()).device}")
    
    # Оптимизатор и loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\n2. Обучение на нескольких батчах...")
    for i in range(5):
        # Генерируем данные
        x = torch.randn(32, 1000, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        
        # Forward
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Батч {i+1}/5: Loss = {loss.item():.4f}")
    
    print("\n✅ Модель успешно обучается на GPU!")


def check_memory_after_operations():
    """Проверяет использование памяти GPU."""
    print("\n" + "=" * 60)
    print("ИСПОЛЬЗОВАНИЕ ПАМЯТИ GPU")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"\n🎮 GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"  Выделено: {memory_allocated:.2f} GB")
        print(f"  Зарезервировано: {memory_reserved:.2f} GB")
        print(f"  Всего: {memory_total:.2f} GB")
        print(f"  Использовано: {(memory_allocated/memory_total)*100:.1f}%")


def main():
    """Главная функция."""
    print("\n" + "🚀" * 30)
    print("ПРОВЕРКА GPU ДЛЯ PYTORCH")
    print("🚀" * 30)
    
    # Информация о PyTorch
    print(f"\n📦 Версия PyTorch: {torch.__version__}")
    
    # Проверка CUDA
    if not check_cuda_available():
        print("\n" + "=" * 60)
        print("❌ ЗАВЕРШЕНИЕ: CUDA недоступна")
        print("=" * 60)
        return
    
    # Информация о CUDA
    check_cuda_info()
    
    # Информация о GPU
    check_gpu_info()
    
    # Тесты
    try:
        test_simple_operation()
        benchmark_speed()
        test_model_training()
        check_memory_after_operations()
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении тестов: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Итоговая информация
    print("\n" + "=" * 60)
    print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 60)
    print("\nВаша RTX 4060 готова к обучению моделей! 🎉")
    print("\nДля запуска обучения:")
    print("  python scripts/train_model.py --model lightgcn --dataset movie_lens")
    print("\nСистема автоматически использует GPU.")
    print("Следите за использованием GPU через: nvidia-smi -l 1")
    print("=" * 60)


if __name__ == "__main__":
    main()


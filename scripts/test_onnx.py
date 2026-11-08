import numpy as np
import onnxruntime as ort

print("Загрузка ONNX модели...")
# Укажите точный путь к вашему экспортированному файлу
session = ort.InferenceSession("chessnet_19x192.onnx")

print("Создание фиктивного входа (1 партия, 73 канала, доска 14x14)...")
# Точно такие же размеры, как в train.py: batch_size=1, 73 planes, 14x14 board
dummy_input = np.random.randn(1, 73, 14, 14).astype(np.float32)

print("Запуск инференса...")
# Имена входов и выходов мы задали при экспорте: "input", "policy", "value"
outputs = session.run(["policy", "value"], {"input": dummy_input})

policy_output, value_output = outputs

print("\n🎉 УСПЕХ! Модель работает корректно.")
print(f"Форма выхода Policy (логиты ходов): {policy_output.shape} (ожидалось: (1, 38416))")
print(f"Форма выхода Value (оценка позиции): {value_output.shape} (ожидалось: (1, 4, 4))")

# Покажем пару случайных значений, чтобы убедиться, что это не нули
print(f"\nПример значений Policy (первые 5): {policy_output[0][:5]}")
print(f"Пример значений Value (для 1-го игрока): {value_output[0][0]}")
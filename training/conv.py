import torch
import torch.onnx
import onnx
from pathlib import Path

def convert_pt_to_onnx(
    pt_model_path: str,
    onnx_model_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 11,
    dynamic_axes: dict = None
):
    """
    Конвертирует PyTorch модель (.pt) в ONNX формат
    
    Args:
        pt_model_path: Путь к .pt файлу
        onnx_model_path: Путь для сохранения .onnx файла
        input_shape: Форма входного тензора (batch, channels, height, width)
        opset_version: Версия ONNX opset
        dynamic_axes: Динамические оси (например, для batch size)
    """
    
    # Загрузка модели
    print(f"Загрузка модели из {pt_model_path}...")
    model = torch.load(pt_model_path, map_location='cpu')
    
    # Если это state_dict, нужно создать модель и загрузить веса
    if isinstance(model, dict) and 'state_dict' in model:
        # Здесь нужно создать архитектуру модели
        # model = YourModelClass()
        # model.load_state_dict(model['state_dict'])
        raise ValueError("Модель содержит state_dict. Создайте архитектуру модели перед загрузкой.")
    
    model.eval()
    
    # Создание dummy input
    dummy_input = torch.randn(input_shape)
    
    # Настройка динамических осей по умолчанию
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Экспорт в ONNX
    print(f"Экспорт модели в ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    # Проверка модели
    print("Проверка ONNX модели...")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ Модель успешно конвертирована в {onnx_model_path}")
    return onnx_model

# Пример использования
if __name__ == "__main__":
    # Для ResNet или похожей модели
    convert_pt_to_onnx(
        pt_model_path="model.pt",
        onnx_model_path="model.onnx",
        input_shape=(1, 3, 224, 224),  # Измените под вашу модель
        opset_version=11
    )
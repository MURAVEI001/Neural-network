from PIL import Image
import numpy as np

def timing_coder(images, time_steps=256, current_strength=10.0):
    """
    Кодирует несколько изображений во временные паттерны спайков.
    
    Параметры:
    - images: numpy array формы (N, H, W) со значениями от 0 до 255,
              либо список PIL Image, либо список путей к файлам.
    - time_steps: количество временных шагов (длительность развёртки).
    - current_strength: величина тока в момент спайка.
    
    Возвращает:
    - numpy array формы (N, num_pixels, time_steps), где для каждого изображения
      и каждого пикселя стоит current_strength в момент спайка, и 0 в остальные моменты.
    """
    # Преобразуем входные данные в единый массив numpy
    if isinstance(images, list):
        # Если список путей или PIL Image
        img_list = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('L')  # загружаем как градации серого
            elif not isinstance(img, np.ndarray):
                img = np.array(img)
            # Приводим к единому размеру (предполагаем, что все изображения одинакового размера)
            img_list.append(np.array(img))
        images = np.array(img_list)
    elif isinstance(images, np.ndarray) and images.ndim == 3:
        pass  # уже готовый массив (N, H, W)
    else:
        raise ValueError("images должен быть списком или 3D массивом (N, H, W)")
    
    N, H, W = images.shape
    num_pixels = H * W
    images_flat = images.reshape(N, num_pixels)
    
    # Создаём выходной массив нулей
    timelines = np.zeros((N, num_pixels, time_steps), dtype=np.float32)
    
    for i in range(N):
        for j in range(num_pixels):
            value = images_flat[i, j]
            # Предполагаем, что значение в диапазоне [0, 255]
            if 0 <= value <= 255:
                # Момент спайка: чем ярче пиксель, тем раньше спайк
                pos = int(255 - value)
                if pos < time_steps:
                    timelines[i, j, pos] = current_strength
                # Если time_steps < 256, можно масштабировать, но здесь просто отбрасываем
    return timelines
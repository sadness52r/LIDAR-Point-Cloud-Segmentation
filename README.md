# LIDAR-Point-Cloud-Segmentation

Короткое руководство по проекту.

**Содержание**
- Быстрый старт
- Запуск для разных датасетов (helimos, helipr, hercules)
- Работа с GPS (структура CSV, визуализация)
- Конфигурация проекта (`src/config.py`)
- Примеры и отладка

---

## Быстрый старт

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Запускайте команды из корневой директории проекта:

```bash
# Пример: визуализация GPS (данные из датасета hercules)
python -m src.app --dataset hercules --gps 03_Day/sensor_data/gps.csv --action map
```

3. После успешного запуска будет создана HTML-карта (по умолчанию `gps_map.html` или путь из конфига).

---

## Запуск для разных датасетов

Доступные датасеты: `helimos`, `helipr`, `hercules`. GPS, INS, IMU и Radar — это сенсорные данные датасета **hercules**.

### HeLiMOS (KITTI-подобный бинарный формат)

```bash
python -m src.app --dataset helimos --bin <path_to_bin> --action cloud
```

### HeLiPR (Aeva LiDAR)

```bash
python -m src.app --dataset helipr --bin <path_to_bin> --action cloud
# график радиальной скорости от азимута
python -m src.app --dataset helipr --bin <path_to_bin> --action velocity
```

### HeRCULES

Датасет hercules содержит несколько типов сенсорных данных. Тип данных определяется аргументом пути:

```bash
# Aeva LiDAR (--bin)
python -m src.app --dataset hercules --bin <path_to_aeva.bin> --action cloud
python -m src.app --dataset hercules --bin <path_to_aeva.bin> --action velocity

# Radar (--radar)
python -m src.app --dataset hercules --radar <path_to_radar.bin> --action cloud
python -m src.app --dataset hercules --radar <path_to_radar.bin> --action velocity

# GPS (--gps)
python -m src.app --dataset hercules --gps <path_to_gps.csv> --action map
python -m src.app --dataset hercules --gps <path_to_gps.csv> --action ego-velocity

# INS (--ins)
python -m src.app --dataset hercules --ins <path_to_ins.csv> --action track
python -m src.app --dataset hercules --ins <path_to_ins.csv> --action ego-velocity

# IMU (--imu)
python -m src.app --dataset hercules --imu <path_to_imu.csv> --action accel
```

Примечание: приложение использует ленивые импорты — heavy-зависимости (например, `open3d`) подгружаются только при визуализации облака точек, поэтому визуализация GPS работает даже без установки всех пакетов.

---

## GPS: структура данных и визуализация

Файл `03_Day/sensor_data/gps.csv` в этом проекте содержит 13 колонок без заголовков. В коде чтения (`src/io/csv_reader.py`) и документации они распарсены как:

0. `timestamp` (нс)
1. `lat` (широта)
2. `lon` (долгота)
3. `height` (высота, м)
4. `velocity_north`
5. `velocity_east`
6. `velocity_up`
7. `roll`
8. `pitch`
9. `azimuth`
10. `status`
11-12. дополнительные/зарезервированные поля

В данном конкретном наборе многие колонки заполнены нулями (статический стенд, или данные отфильтрованы). Для более точной информации смотрите `03_Day/sensor_data/inspva.csv` — там есть исчерпывающие INS/IMU-значения (roll/pitch/azimuth и компоненты скорости).

Визуализация:
- Скрипт сохраняет интерактивную карту Folium (HTML). Открыть её можно:
	- через двойной клик в проводнике или в VS Code
	- через локальный сервер: `python -m http.server 8000` и перейти на `http://localhost:8000/gps_map.html`

---

## Конфигурация проекта — `src/config.py`

Файл `src/config.py` содержит все константы путей и параметры визуализации/обработки:
- Пути к данным (DATA_DIR, SENSOR_DATA_DIR, GPS_DATA_FILE и т.д.)
- Пути к результатам (GPS_MAP_FILE, GPS_TRAJECTORY_MAP_FILE и т.д.)
- Параметры визуализации (MAP_ZOOM_LEVEL, POLYLINE_COLOR...)
- Параметры обработки (POINT_SKIP, GPS_SKIP, FLOAT_PRECISION)

Если хотите изменить параметры — правьте `src/config.py` или переопределяйте значения в рантайме.

---

## Motion Object Segmentation (MOS)

Модуль сегментации движущихся объектов в облаке точек LiDAR. Классифицирует каждую точку как **static** или **moving** с помощью Random Forest, обученного на размеченных данных HeLiMOS.

### 1. Обучение модели

Обучение на HeLiMOS (ground truth labels: static=9, moving=251):

```bash
# CPU (Random Forest, scikit-learn)
python -m src.app --dataset helimos --action mos-train --sequence data/Deskewed_LiDAR --sensor Velodyne --split train

# GPU (XGBoost CUDA) — значительно быстрее при наличии NVIDIA GPU
python -m src.app --dataset helimos --action mos-train --sequence data/Deskewed_LiDAR --sensor Velodyne --split train --gpu
```

Параметры:
- `--sequence` — путь к корню датасета Deskewed_LiDAR (по умолчанию: `data/Deskewed_LiDAR`)
- `--sensor` — тип сенсора: `Velodyne`, `Ouster`, `Avia`, `Aeva` (по умолчанию: `Velodyne`)
- `--split` — раздел датасета: `train`, `val`, `test` (по умолчанию: `train`)
- `--max-frames` — ограничить число кадров для обучения (по умолчанию: все)
- `--model` — путь для сохранения модели (по умолчанию: `models/mos_rf.pkl`)
- `--threshold` — порог P(moving) для RF классификатора (по умолчанию: `0.85`)
- `--inlier-threshold` — RANSAC inlier threshold в м/с для Doppler-сенсоров (по умолчанию: `0.5`)
- `--gpu` — обучение на GPU через XGBoost CUDA (требуется NVIDIA GPU и установленный `xgboost`)

Модель сохраняется в `models/mos_rf.pkl`. Модели GPU и CPU взаимозаменяемы при инференсе.

### 2. Инференс на одном кадре

Работает с любым датасетом — loader выбирается по `--dataset`.

**Важно:** для сенсоров с Doppler-скоростью (Aeva, Radar) используется RANSAC — обученная модель **не нужна**, `--model` можно не указывать. Для сенсоров без Допплера (Velodyne, Ouster) требуется обученная модель (`--model`).

```bash
# HeLiMOS Velodyne (без Допплера — нужна модель)
python -m src.app --dataset helimos --bin <path_to_helimos.bin> --action mos --model models/mos_rf.pkl

# HeLiPR Aeva (есть Допплер — модель не нужна, используется RANSAC)
python -m src.app --dataset helipr --bin <path_to_aeva.bin> --action mos

# Hercules Aeva (есть Допплер — RANSAC)
python -m src.app --dataset hercules --bin <path_to_aeva.bin> --action mos

# Hercules Radar (есть Допплер — RANSAC)
python -m src.app --dataset hercules --radar <path_to_radar.bin> --action mos
```

С изображением стерео-камеры (файлы из `03_Day/stereo_left`):

```bash
python -m src.app --dataset hercules --bin 03_Day/Aeva/1738300381892782006.bin --action mos --camera 03_Day/stereo_left
```

Ближайший по временной метке кадр камеры автоматически подбирается и отображается на нижней панели графика.

Результат — графики matplotlib:
- **Левый**: radial velocity vs azimuth (если есть Допплер) — серые точки = static, красные = moving, чёрная линия = RANSAC-кривая эго-движения
- **Правый**: bird's-eye view (x, y) — те же цвета
- **Нижний** (при `--camera`): изображение со стерео-камеры

Параметры MOS:
- `--threshold` — порог P(moving) для RF классификатора (по умолчанию: `0.85`)
- `--inlier-threshold` — RANSAC inlier threshold в м/с для Doppler-сенсоров (по умолчанию: `0.5`). Увеличение значения снижает число ложных moving-точек

### 3. Инференс на последовательности кадров

Использует temporal consistency через SE(3) позы для более точной сегментации:

```bash
python -m src.app --dataset helimos --action mos-sequence --sequence data/Deskewed_LiDAR --sensor Velodyne --split val --n-frames 10 --n-context 3 --model models/mos_rf.pkl
```

Параметры:
- `--n-frames` — число кадров для обработки (по умолчанию: `5`)
- `--n-context` — размер временного окна для temporal consistency (по умолчанию: `3`)

### 4. Покадровый рендеринг последовательности (для GIF / видео)

Скрипт `src/examples/mos_sequence_example.py` обрабатывает все кадры Aeva и сохраняет каждый как PNG с фиксированными осями (графики не скачут между кадрами):

```bash
python -m src.examples.mos_sequence_example \
    --aeva   03_Day/Aeva \
    --camera 03_Day/stereo_left \
    --output output/mos_frames
```

Для Aeva (есть Допплер) модель не нужна — используется RANSAC. Для сенсоров без Допплера добавьте `--model models/mos_rf.pkl`.

Параметры:
- `--start` — начальный индекс кадра (по умолчанию: `0`)
- `--max-frames` — максимальное число кадров (по умолчанию: все)
- `--model` — путь к MOS модели (нужна только для сенсоров без Допплера)
- `--inlier-threshold` — RANSAC порог (по умолчанию: `0.5`)
- `--dpi` — разрешение выходных изображений (по умолчанию: `120`)

Сетка 2x2: камера (верх-лево), BEV полный (верх-право), velocity vs azimuth (низ-лево), BEV зум (низ-право).

Сборка GIF/видео из полученных кадров:

```bash
# GIF (ImageMagick)
magick convert -delay 10 -loop 0 output/mos_frames/*.png mos.gif

# MP4 (ffmpeg)
ffmpeg -framerate 10 -i output/mos_frames/%06d.png -c:v libx264 -pix_fmt yuv420p mos.mp4
```

---

## Примеры

- `src/examples/gps_example.py` — чтение и визуализация GPS на карте (использует константы из `src/config.py`)
- `src/examples/mos_sequence_example.py` — покадровый рендеринг MOS для Hercules Aeva + стерео-камера
- CLI: `src/app.py` — универсальный интерфейс для всех датасетов

---

## Зависимости

См. `requirements.txt`. Основные пакеты:
- pandas, numpy — базовая работа с данными
- pyproj - преобразование координат GPS 
- folium — интерактивные карты
- matplotlib — графики
- open3d — (опционально) визуализация облаков точек
- scikit-learn — Random Forest для MOS
- xgboost — (опционально) GPU-ускоренное обучение MOS через CUDA
- joblib — сериализация модели

Примечание: если вам не нужна визуализация облаков точек, можно пропустить установку `open3d`. `xgboost` нужен только при использовании флага `--gpu`.


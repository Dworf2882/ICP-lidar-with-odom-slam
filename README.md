# Документация по NDT SLAM Node

## 1. Обзор

NDT SLAM (Normal Distributions Transform Simultaneous Localization and Mapping) - это алгоритм одновременной локализации и построения карты, который использует статистическое представление пространства для регистрации лидарных сканов. 

**Текущая реализация**: представляет собой упрощенную версию алгоритма, использующую ICP (Iterative Closest Point) как временное решение вместо полноценного NDT.

## 2. Математические основы

### 2.1. Системы координат и преобразования

Робот оперирует в нескольких системах координат:
- **odom**: система одометрии (накапливающая ошибку)
- **map**: глобальная система координат карты  
- **base_link**: система координат робота

Преобразование между системами описывается матрицами 4×4 в однородных координатах:

$$
T = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
$$

где:
- $R$ - матрица поворота 3×3
- $t$ - вектор переноса 3×1

### 2.2. Преобразование кватернионов

Используется преобразование между кватернионами и матрицами поворота:

```python
# Из кватерниона в матрицу
rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

# Из матрицы в кватернион
quaternion = R.from_matrix(rotation_matrix).as_quat()
```

### 2.3. Преобразование лидарных данных

**Преобразование LaserScan в облако точек**:

```python
def laser_scan_to_pointcloud(self, scan_msg):
    points = []
    angle = scan_msg.angle_min
    
    for i, range_val in enumerate(scan_msg.ranges):
        if scan_msg.range_min <= range_val <= scan_msg.range_max:
            # Преобразование из полярных в декартовы координаты
            x = range_val * math.cos(angle)  # x = r * cos(θ)
            y = range_val * math.sin(angle)  # y = r * sin(θ)
            z = 0.0
            points.append([x, y, z])
        angle += scan_msg.angle_increment
    
    return np.array(points) if points else np.empty((0, 3))
```

**Математическое обоснование**: Используется стандартное преобразование из полярных координат $(r, \theta)$ в декартовы $(x, y)$:

$$
\begin{aligned}
x &= r \cdot \cos(\theta) \\
y &= r \cdot \sin(\theta) \\
z &= 0
\end{aligned}
$$

### 2.4. Воксельный даунсемплинг

```python
def voxel_downsample(self, points, voxel_size):
    if len(points) == 0:
        return points
        
    # Вычисляем воксельные индексы
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Находим уникальные воксели
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    return points[unique_indices]
```

**Математическое обоснование**: Пространство делится на воксели размера $\text{voxel\_size}$. Все точки внутри одного вокселя заменяются одной представительной точкой.

Воксельный индекс вычисляется как:
$$
\text{index} = \left\lfloor \frac{\text{point}}{\text{voxel\_size}} \right\rfloor
$$

## 3. Архитектура алгоритма

### 3.1. Основной цикл обработки

```
Лидарный скан → Преобразование в точки → Предобработка → Регистрация → Обновление карты
       ↑
Одометрия → Предсказание позы → Уточнение позы
```

### 3.2. Детальное описание компонентов

#### Преобразование одометрии

```python
def odom_to_transform(self, odom_msg):
    pose = odom_msg.pose.pose
    
    # Позиция и ориентация
    x, y, z = pose.position.x, pose.position.y, pose.position.z
    qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    
    # Создание матрицы преобразования
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    transform[0, 3], transform[1, 3], transform[2, 3] = x, y, z
    
    return transform
```

**Математическое обоснование**: Кватернион преобразуется в матрицу поворота 3×3, которая комбинируется с вектором переноса:

$$
T = \begin{bmatrix}
R_{3\times3} & t_{3\times1} \\
0_{1\times3} & 1
\end{bmatrix}
$$

#### Алгоритм ICP регистрации

```python
def simple_icp_registration(self, source_points, target_points, initial_transform, max_iterations=20):
    """
    Упрощенная ICP регистрация
    
    Алгоритм минимизирует целевую функцию:
    E(R,t) = Σ||(R·p_i + t) - q_i||²
    """
    if len(source_points) == 0 or len(target_points) == 0:
        return initial_transform, 0.0
        
    current_transform = initial_transform.copy()
    
    for iteration in range(max_iterations):
        # 1. Преобразуем source points
        transformed_source = self.transform_points(source_points, current_transform)
        
        # 2. Поиск соответствий
        tree = KDTree(target_points)
        distances, indices = tree.query(transformed_source)
        
        # 3. Отбрасывание выбросов
        valid_mask = distances < self.max_correspondence_distance
        if np.sum(valid_mask) < 10:
            break
            
        valid_source = transformed_source[valid_mask]
        valid_target = target_points[indices[valid_mask]]
        
        # 4. Вычисление оптимального преобразования
        source_mean = np.mean(valid_source, axis=0)
        target_mean = np.mean(valid_target, axis=0)
        
        centered_source = valid_source - source_mean
        centered_target = valid_target - target_mean
        
        # Матрица ковариации H = Σ(p̂_i · q̂_iᵀ)
        H = centered_source.T @ centered_target
        
        # SVD разложение
        U, S, Vt = np.linalg.svd(H)
        
        # Оптимальная матрица поворота
        rotation = Vt.T @ U.T
        
        # Коррекция для отражений
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T
        
        # Оптимальный перенос
        translation = target_mean - rotation @ source_mean
        
        # Обновление преобразования
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = rotation
        delta_transform[:3, 3] = translation
        
        current_transform = delta_transform @ current_transform
        
        # Проверка сходимости
        if iteration > 0 and np.max(np.abs(translation)) < 1e-6:
            break
    
    # Оценка качества регистрации
    fitness = np.sum(valid_mask) / len(source_points) if len(source_points) > 0 else 0.0
    
    return current_transform, fitness
```

**Математическое обоснование ICP**:

1. **Целевая функция**: 
$$
\min_{R,t} \sum \| (R \cdot p_i + t) - q_i \|^2
$$

2. **Оптимальное преобразование** находится через SVD разложение матрицы ковариации:
   - $H = \sum (\hat{p}_i \cdot \hat{q}_i^T)$ где $\hat{p}_i = p_i - \mu_p$, $\hat{q}_i = q_i - \mu_q$
   - $R = V \cdot U^T$ из $\text{SVD}(H) = U \cdot \Sigma \cdot V^T$
   - $t = \mu_q - R \cdot \mu_p$

3. **Сходимость**: Алгоритм итеративно уменьшает среднеквадратичную ошибку.

#### Процесс SLAM

```python
def process_scan(self, scan_points, odom_transform, stamp):
    """Основной процесс SLAM"""
    
    # Предсказание позы на основе одометрии
    if self.last_processed_odom is not None:
        relative_transform = np.linalg.inv(self.last_processed_odom) @ odom_transform
    else:
        relative_transform = np.eye(4)
    
    predicted_pose = self.current_pose @ relative_transform
    
    # Измерение: уточнение позы через регистрацию
    corrected_pose, fitness = self.simple_icp_registration(
        scan_points, self.global_map_points, predicted_pose
    )
    
    if fitness > 0.1:
        self.current_pose = corrected_pose
        
        # Обновление карты
        scan_global = self.transform_points(scan_points, self.current_pose)
        self.update_global_map(scan_global)
```

**Математическое обоснование SLAM**:

1. **Предсказание**: 
$$
\hat{x}_k = f(x_{k-1}, u_k) + w_k
$$
где $x$ - поза, $u$ - управление (одометрия), $w$ - шум

2. **Коррекция**:
$$
x_k = \arg\min_x \sum \| h(x, z_i) - m_j \|^2
$$
где $z$ - измерения лидара, $m$ - карта

3. **Обновление карты**:
$$
M_k = M_{k-1} \cup \{ T(x_k) \cdot z_i \}
$$

## 4. Параметры и настройка

### 4.1. Ключевые параметры ROS 2

```python
# Параметры, объявленные в коде
self.declare_parameter('voxel_size', 0.3)
self.declare_parameter('max_correspondence_distance', 2.0)
self.declare_parameter('resolution', 1.0)
self.declare_parameter('scan_queue_size', 10)
self.declare_parameter('map_publish_rate', 1.0)
```

### 4.2. Детальное описание параметров

#### 1. **`voxel_size` (по умолчанию: 0.3)**
- **Назначение**: Размер вокселя для даунсемплинга облака точек
- **Влияние**: 
  - Меньшие значения → более детальная карта, но выше вычислительная нагрузка
  - Большие значения → меньше деталей, но выше производительность
- **Рекомендации**:
  - Для помещений: 0.1-0.2 м
  - Для наружных сред: 0.3-0.5 м

#### 2. **`max_correspondence_distance` (по умолчанию: 2.0)**
- **Назначение**: Максимальное расстояние для поиска соответствий в ICP
- **Влияние**:
  - Меньшие значения → более точные соответствия
  - Большие значения → устойчивее к большим смещениям
- **Рекомендации**:
  - Зависит от скорости движения робота
  - Обычно 1.5-3.0 м

#### 3. **`resolution` (по умолчанию: 1.0)**
- **Назначение**: Разрешение карты
- **Влияние**:
  - Меньшие значения → более высокая детализация карты
  - Большие значения → меньше использование памяти
- **Рекомендации**:
  - Для точной навигации: 0.5-1.0 м
  - Для грубого картирования: 1.0-2.0 м

#### 4. **`scan_queue_size` (по умолчанию: 10)**
- **Назначение**: Размер очереди сканов для обработки
- **Влияние**:
  - Меньшие значения → меньшая задержка
  - Большие значения → буферизация данных
- **Рекомендации**:
  - Для высокочастотных лидаров: 10-20
  - Для низкочастотных лидаров: 5-10

#### 5. **`map_publish_rate` (по умолчанию: 1.0)**
- **Назначение**: Частота публикации карты (Гц)
- **Влияние**:
  - Меньшие значения → меньше нагрузки на сеть
  - Большие значения → более актуальная карта
- **Рекомендации**:
  - Для статических карт: 0.5-1.0 Гц
  - Для динамических сред: 1.0-2.0 Гц

### 4.3. Дополнительные параметры в коде

6. **Минимальное количество точек в скане** (`100`)
   - Отбрасывает сканы с меньшим количеством точек

7. **Минимальное количество соответствий ICP** (`10`)
   - Минимальное количество валидных соответствий для продолжения ICP

8. **Максимальное количество итераций ICP** (`20`)
   - Ограничивает время выполнения регистрации

9. **Порог fitness для обновления карты** (`0.1`)
   - Минимальное качество регистрации для принятия результата

10. **Максимальный размер карты** (`10000` точек)
    - Автоматический даунсемплинг при превышении этого размера

### 4.4. Рекомендации по настройке для различных сценариев

#### Для помещений (indoors)
```yaml
voxel_size: 0.1
max_correspondence_distance: 1.0
resolution: 0.3
scan_queue_size: 10
map_publish_rate: 2.0
```

#### Для наружных сред (outdoors)
```yaml
voxel_size: 0.5
max_correspondence_distance: 3.0
resolution: 1.0
scan_queue_size: 5
map_publish_rate: 1.0
```

#### Для высокоточного картирования
```yaml
voxel_size: 0.05
max_correspondence_distance: 0.5
resolution: 0.1
scan_queue_size: 20
map_publish_rate: 1.0
```

#### Для быстрой навигации
```yaml
voxel_size: 0.3
max_correspondence_distance: 2.0
resolution: 0.8
scan_queue_size: 5
map_publish_rate: 0.5
```

## 5. Интерфейсы ROS 2

### 5.1. Подписки (Subscriptions)

- **`/scan`** (`sensor_msgs/LaserScan`)
  - Входные данные лидара
  - Содержит информацию о расстояниях и углах сканирования

- **`/odom`** (`nav_msgs/Odometry`)
  - Одометрия робота
  - Используется для предсказания движения между сканами

### 5.2. Публикации (Publications)

- **`/map`** (`nav_msgs/OccupancyGrid`)
  - Построенная карта в формате occupancy grid
  - Публикуется с частотой `map_publish_rate`

- **`/map_points`** (`sensor_msgs/PointCloud2`)
  - Облако точек глобальной карты
  - Используется для визуализации и отладки

- **`/tf`** (`tf2_msgs/TFMessage`)
  - Трансформы между системами координат
  - Включает преобразования `map` → `odom` → `base_link`

## 6. Ограничения и улучшения

### 6.1. Текущие ограничения

1. **Используется упрощенный ICP вместо полноценного NDT**
   - NDT обеспечивает более устойчивую регистрацию при наличии шума
   - Требует вычисления нормальных распределений для вокселей

2. **Нет оптимизации графа поз**
   - Отсутствует глобальная оптимизация при обнаружении петель
   - Накопление ошибки со временем

3. **Отсутствует обработка выбросов и динамических объектов**
   - Динамические объекты могут искажать карту
   - Нет механизма фильтрации ложных соответствий

### 6.2. Возможные улучшения

1. **Реализация полноценного NDT**
   - Использование нормальных распределений для представления вокселей
   - Более устойчивая к шуму регистрация

2. **Loop closure detection**
   - Добавление обнаружения петель
   - Глобальная оптимизация графа поз

3. **Адаптивные параметры**
   - Автоматическая настройка параметров в зависимости от характеристик среды
   - Динамическое изменение `max_correspondence_distance` на основе скорости

4. **Многопоточная обработка**
   - Выделение регистрации в отдельный поток
   - Асинхронное обновление карты

## 7. Производительность и оптимизация

### 7.1. Вычислительная сложность

- **ICP регистрация**: O(n·log(m)) где n - точки скана, m - точки карты
- **Построение KDTree**: O(m·log(m))
- **Воксельный даунсемплинг**: O(n)

### 7.2. Оптимизационные стратегии

1. **Агрессивный даунсемплинг** для больших карт
2. **Ограничение размера карты** с сохранением ключевых областей
3. **Использование приближенных nearest neighbor поисков**
4. **Пропуск кадров** при высокой нагрузке

## 8. Заключение

Данная реализация предоставляет базовый фреймворк для 2D SLAM с использованием лидара. Математическая основа включает преобразования систем координат, ICP регистрацию и байесовскую фильтрацию для совместной локализации и построения карты. 

Алгоритм эффективен для средних по размеру помещений и наружных сред, но требует доработок для работы в крупномасштабных или высокодинамичных окружениях. Ключевые направления улучшения включают реализацию полноценного NDT, добавление loop closure и адаптивных параметров.

Для достижения наилучших результатов рекомендуется тщательная настройка параметров под конкретную среду эксплуатации и характеристики сенсоров.

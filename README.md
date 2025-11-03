# Документация по NDT SLAM Node

## Обзор

NDT SLAM (Normal Distributions Transform Simultaneous Localization and Mapping) - это алгоритм одновременной локализации и построения карты, который использует статистическое представление пространства для регистрации лидарных сканов. Данная реализация представляет собой упрощенную версию алгоритма, использующую ICP (Iterative Closest Point).

## Принципы работы

### 1. Математические основы

#### Преобразование систем координат

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

#### Кватернионы и матрицы поворота

Используется преобразование между кватернионами и матрицами поворота:

```python
# Из кватерниона в матрицу
rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

# Из матрицы в кватернион
quaternion = R.from_matrix(rotation_matrix).as_quat()
```

### 2. Архитектура алгоритма

#### Основной цикл обработки

```
Лидарный скан → Преобразование в точки → Предобработка → Регистрация → Обновление карты
       ↑
Одометрия → Предсказание позы → Уточнение позы
```

## 3. Детальное описание компонентов


### Преобразование LaserScan в облако точек

```python
def laser_scan_to_pointcloud(self, scan_msg):
    """Преобразование LaserScan в numpy массив точек с математическим обоснованием"""
    points = []
    angle = scan_msg.angle_min
    
    for i, range_val in enumerate(scan_msg.ranges):
        if scan_msg.range_min <= range_val <= scan_msg.range_max:
            # Преобразование из полярных в декартовы координаты
            # x = r * cos(θ)
            # y = r * sin(θ)  
            # z = 0 (для 2D лидара)
            x = range_val * math.cos(angle)
            y = range_val * math.sin(angle)
            z = 0.0
            points.append([x, y, z])
        angle += scan_msg.angle_increment
    
    return np.array(points) if points else np.empty((0, 3))
```

**Математическое обоснование**: Используется стандартное преобразование из полярных координат $(r, \theta)$ в декартовы $(x, y)$ для каждого луча лидара:

$$
\begin{aligned}
x &= r \cdot \cos(\theta) \\
y &= r \cdot \sin(\theta) \\
z &= 0
\end{aligned}
$$

### Воксельный даунсемплинг

```python
def voxel_downsample(self, points, voxel_size):
    """Простой воксельный даунсемплинг с математическим обоснованием"""
    if len(points) == 0:
        return points
        
    # Вычисляем воксельные индексы: floor(points / voxel_size)
    # Каждая точка попадает в воксель с целочисленными координатами
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Находим уникальные воксели и берем по одной точке из каждого
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    return points[unique_indices]
```

**Математическое обоснование**: Пространство делится на воксели размера ${voxel\_size}$. Все точки внутри одного вокселя заменяются одной представительной точкой. Это уменьшает вычислительную сложность и устраняет избыточность данных.

Воксельный индекс вычисляется как:
$$
{index} = \left\lfloor \frac{\text{point}}{\text{voxel\_size}} \right\rfloor
$$

### Преобразование одометрии

```python
def odom_to_transform(self, odom_msg):
    """Преобразование Odometry в матрицу преобразования 4x4 с математическим обоснованием"""
    pose = odom_msg.pose.pose
    
    # Позиция
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    
    # Ориентация (кватернион)
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w
    
    # Создание матрицы преобразования
    # T = [ R | t ]
    #     [ 0 | 1 ]
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    transform[0, 3] = x
    transform[1, 3] = y
    transform[2, 3] = z
    
    return transform
```

**Математическое обоснование**: Кватернион преобразуется в матрицу поворота 3×3, которая комбинируется с вектором переноса для формирования матрицы преобразования 4×4 в однородных координатах:

$$
T = \begin{bmatrix}
R_{3\times3} & t_{3\times1} \\
0_{1\times3} & 1
\end{bmatrix}
$$

### Алгоритм ICP регистрации

```python
def simple_icp_registration(self, source_points, target_points, initial_transform, max_iterations=20):
    """
    Упрощенная ICP регистрация с математическим обоснованием
    
    Алгоритм минимизирует целевую функцию:
    E(R,t) = Σ||(R·p_i + t) - q_i||²
    где p_i - точки source, q_i - соответствующие точки target
    """
    if len(source_points) == 0 or len(target_points) == 0:
        return initial_transform, 0.0
        
    current_transform = initial_transform.copy()
    
    for iteration in range(max_iterations):
        # 1. Преобразуем source points: p_i' = R·p_i + t
        transformed_source = self.transform_points(source_points, current_transform)
        
        # 2. Поиск соответствий: для каждой p_i' найти ближайшую q_i
        tree = KDTree(target_points)
        distances, indices = tree.query(transformed_source)
        
        # 3. Отбрасывание выбросов по порогу расстояния
        valid_mask = distances < self.max_correspondence_distance
        if np.sum(valid_mask) < 10:
            break
            
        valid_source = transformed_source[valid_mask]
        valid_target = target_points[indices[valid_mask]]
        
        # 4. Вычисление оптимального преобразования
        # Центрирование точек
        source_mean = np.mean(valid_source, axis=0)
        target_mean = np.mean(valid_target, axis=0)
        
        centered_source = valid_source - source_mean
        centered_target = valid_target - target_mean
        
        # Матрица ковариации H = Σ(p̂_i · q̂_iᵀ)
        H = centered_source.T @ centered_target
        
        # SVD разложение: H = U·Σ·Vᵀ
        U, S, Vt = np.linalg.svd(H)
        
        # Оптимальная матрица поворота: R = V·Uᵀ
        rotation = Vt.T @ U.T
        
        # Коррекция для отражений (det(R) должен быть +1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T
        
        # Оптимальный перенос: t = μ_q - R·μ_p
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

3. **Сходимость**: Алгоритм итеративно уменьшает среднеквадратичную ошибку до достижения сходимости.

### Процесс SLAM

```python
def process_scan(self, scan_points, odom_transform, stamp):
    """Основной процесс SLAM с математическим обоснованием"""
    
    # Предсказание позы на основе одометрии
    # T_predicted = T_previous · ΔT_odom
    if self.last_processed_odom is not None:
        relative_transform = np.linalg.inv(self.last_processed_odom) @ odom_transform
    else:
        relative_transform = np.eye(4)
    
    predicted_pose = self.current_pose @ relative_transform
    
    # Измерение: уточнение позы через регистрацию
    # T_corrected = argmin_T Σ||T·p_i - m_j||²
    corrected_pose, fitness = self.simple_icp_registration(
        scan_points, self.global_map_points, predicted_pose
    )
    
    if fitness > 0.1:
        self.current_pose = corrected_pose
        
        # Обновление карты: m_new = T_corrected · p_scan
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

## Параметры и настройка

### Ключевые параметры:

- `voxel_size` (0.3): Размер вокселя для даунсемплинга
- `max_correspondence_distance` (2.0): Максимальное расстояние для соответствий в ICP
- `resolution` (1.0): Разрешение карты
- `scan_queue_size` (10): Размер очереди сканов

### Рекомендации по настройке:

1. **Для помещений**: уменьшить `voxel_size` до 0.1-0.2
2. **Для больших пространств**: увеличить `max_correspondence_distance`
3. **Для точной навигации**: уменьшить `voxel_size` и увеличить `resolution`

## Ограничения и улучшения

### Текущие ограничения:

1. Используется упрощенный ICP вместо полноценного NDT
2. Нет оптимизации графа поз
3. Отсутствует обработка выбросов и динамических объектов

### Возможные улучшения:

1. **Реализация NDT**: Использование нормальных распределений для представления вокселей
2. **Loop closure**: Добавление обнаружения петель и глобальной оптимизации
3. **Адаптивные параметры**: Автоматическая настройка параметров характеристик среды

## Заключение

Данная реализация предоставляет базовый для 2D SLAM с использованием лидара. Математическая основа включает преобразования систем координат, ICP регистрацию и байесовскую фильтрацию для совместной локализации и построения карты. Алгоритм может быть улучшен добавлением более сложных методов регистрации и оптимизации.

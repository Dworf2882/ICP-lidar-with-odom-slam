#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from tf2_ros import TransformBroadcaster
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import math
from collections import deque
import threading
import std_msgs.msg

class NdtSlamNode(Node):
    def __init__(self):
        super().__init__('ndt_slam_node')
        
        # Параметры
        self.declare_parameter('voxel_size', 0.3)
        self.declare_parameter('max_correspondence_distance', 2.0)
        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('scan_queue_size', 10)
        self.declare_parameter('map_publish_rate', 1.0)
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.max_correspondence_distance = self.get_parameter('max_correspondence_distance').value
        self.resolution = self.get_parameter('resolution').value
        self.scan_queue_size = self.get_parameter('scan_queue_size').value
        
        # Переменные состояния
        self.current_pose = np.eye(4)  # Текущая поза в системе координат карты
        self.global_map_points = np.empty((0, 3))  # Глобальная карта как numpy array
        self.last_odom = None
        self.odom_transform = np.eye(4)
        self.map_initialized = False
        self.last_processed_odom = None
        
        # Потокобезопасность
        self.lock = threading.Lock()
        
        # Подписки
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Публикации
        self.pose_pub = self.create_publisher(PoseStamped, '/ndt_pose', 10)
        self.map_pub = self.create_publisher(PointCloud2, '/ndt_map', 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Таймер для публикации карты
        self.map_timer = self.create_timer(
            1.0 / self.get_parameter('map_publish_rate').value,
            self.publish_map
        )
        
        self.get_logger().info('NDT SLAM node initialized')

    def laser_scan_to_pointcloud(self, scan_msg):
        """Преобразование LaserScan в numpy массив точек"""
        points = []
        angle = scan_msg.angle_min
        
        for i, range_val in enumerate(scan_msg.ranges):
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                # Преобразование из полярных в декартовы координаты
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                z = 0.0  # 2D лидар
                points.append([x, y, z])
            angle += scan_msg.angle_increment
        
        return np.array(points) if points else np.empty((0, 3))

    def voxel_downsample(self, points, voxel_size):
        """Простой воксельный даунсемплинг"""
        if len(points) == 0:
            return points
            
        # Вычисляем воксельные индексы
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Находим уникальные воксели
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        
        return points[unique_indices]

    def preprocess_pointcloud(self, points):
        """Предобработка облака точек"""
        if len(points) == 0:
            return points
            
        # Даунсемплинг
        points_down = self.voxel_downsample(points, self.voxel_size)
        
        return points_down

    def odom_to_transform(self, odom_msg):
        """Преобразование Odometry в матрицу преобразования 4x4"""
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
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        transform[0, 3] = x
        transform[1, 3] = y
        transform[2, 3] = z
        
        return transform

    def transform_points(self, points, transformation):
        """Применение преобразования к точкам"""
        if len(points) == 0:
            return points
            
        # Преобразуем в homogeneous coordinates
        homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
        transformed_points = (transformation @ homogeneous_points.T).T
        return transformed_points[:, :3]

    def simple_icp_registration(self, source_points, target_points, initial_transform, max_iterations=20):
        """
        Упрощенная ICP регистрация вместо NDT
        Это временное решение пока не починим NDT
        """
        if len(source_points) == 0 or len(target_points) == 0:
            result = type('Result', (), {})()
            result.transformation = initial_transform
            result.fitness = 0.0
            return result
            
        current_transform = initial_transform.copy()
        prev_error = float('inf')
        
        for iteration in range(max_iterations):
            # Преобразуем source points
            transformed_source = self.transform_points(source_points, current_transform)
            
            # Находим ближайшие точки (простой вариант)
            from scipy.spatial import KDTree
            tree = KDTree(target_points)
            distances, indices = tree.query(transformed_source)
            
            # Фильтруем слишком далекие точки
            valid_mask = distances < self.max_correspondence_distance
            if np.sum(valid_mask) < 10:  # Минимум соответствий
                break
                
            valid_source = transformed_source[valid_mask]
            valid_target = target_points[indices[valid_mask]]
            
            # Вычисляем среднее преобразование
            source_mean = np.mean(valid_source, axis=0)
            target_mean = np.mean(valid_target, axis=0)
            
            # Центрируем точки
            centered_source = valid_source - source_mean
            centered_target = valid_target - target_mean
            
            # Вычисляем rotation matrix используя SVD
            H = centered_source.T @ centered_target
            U, S, Vt = np.linalg.svd(H)
            rotation = Vt.T @ U.T
            
            # Убедимся что rotation matrix (det = 1)
            if np.linalg.det(rotation) < 0:
                Vt[-1, :] *= -1
                rotation = Vt.T @ U.T
            
            # Вычисляем translation
            translation = target_mean - rotation @ source_mean
            
            # Обновляем преобразование
            delta_transform = np.eye(4)
            delta_transform[:3, :3] = rotation
            delta_transform[:3, 3] = translation
            
            current_transform = delta_transform @ current_transform
            
            # Проверяем сходимость
            mean_error = np.mean(distances[valid_mask])
            if abs(prev_error - mean_error) < 1e-6:
                break
            prev_error = mean_error
        
        result = type('Result', (), {})()
        result.transformation = current_transform
        # Простая оценка fitness - доля валидных соответствий
        result.fitness = np.sum(valid_mask) / len(source_points) if len(source_points) > 0 else 0.0
        
        return result

    def odom_callback(self, msg):
        """Обработка одометрии"""
        with self.lock:
            self.last_odom = msg
            self.odom_transform = self.odom_to_transform(msg)

    def scan_callback(self, msg):
        """Обработка данных лидара"""
        if self.last_odom is None:
            self.get_logger().warn('No odometry data received yet')
            return
        
        try:
            # Преобразование скана в точки
            scan_points = self.laser_scan_to_pointcloud(msg)
            
            if len(scan_points) < 100:  # Минимальное количество точек
                self.get_logger().warn('Too few points in scan')
                return
            
            # Предобработка
            scan_points_down = self.preprocess_pointcloud(scan_points)
            
            with self.lock:
                current_odom_transform = self.odom_transform.copy()
                last_odom = copy.deepcopy(self.last_odom)
            
            # Обработка в отдельном потоке для производительности
            threading.Thread(
                target=self.process_scan,
                args=(scan_points_down, current_odom_transform, msg.header.stamp),
                daemon=True
            ).start()
            
        except Exception as e:
            self.get_logger().error(f'Error processing scan: {str(e)}')

    def process_scan(self, scan_points, odom_transform, stamp):
        """Обработка скана и обновление карты"""
        try:
            if not self.map_initialized:
                # Инициализация карты первым сканом
                with self.lock:
                    self.global_map_points = scan_points
                    self.current_pose = odom_transform
                    self.map_initialized = True
                
                self.get_logger().info('Global map initialized with first scan')
                return
            
            # Получаем относительное преобразование из одометрии
            if self.last_processed_odom is not None:
                relative_transform = np.linalg.inv(self.last_processed_odom) @ odom_transform
            else:
                relative_transform = np.eye(4)
            
            # Предсказание текущей позы
            predicted_pose = self.current_pose @ relative_transform
            
            # Предобработка глобальной карты
            with self.lock:
                global_map_down = self.voxel_downsample(self.global_map_points, self.voxel_size)
            
            # Регистрация (используем упрощенный ICP вместо NDT)
            result = self.simple_icp_registration(scan_points, global_map_down, predicted_pose)
            
            if result.fitness > 0.1:  # Минимальный порог качества
                # Обновление позы
                self.current_pose = result.transformation
                
                # Преобразование текущего скана в глобальную систему координат
                scan_transformed = self.transform_points(scan_points, self.current_pose)
                
                # Обновление глобальной карты
                with self.lock:
                    # Объединение с существующей картой
                    if len(self.global_map_points) > 0:
                        self.global_map_points = np.vstack([self.global_map_points, scan_transformed])
                    else:
                        self.global_map_points = scan_transformed
                    
                    # Даунсемплинг для контроля размера
                    if len(self.global_map_points) > 10000:
                        self.global_map_points = self.voxel_downsample(self.global_map_points, self.voxel_size)
            
            # Сохраняем одометрию для следующего шага
            self.last_processed_odom = odom_transform
            
            # Публикация результатов
            self.publish_pose(stamp)
            self.publish_tf(stamp)
            
            self.get_logger().info(f'Registration completed. Fitness: {result.fitness:.3f}')
            
        except Exception as e:
            self.get_logger().error(f'Error in process_scan: {str(e)}')

    def publish_pose(self, stamp):
        """Публикация позы"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        
        # Извлечение позиции
        pose_msg.pose.position.x = self.current_pose[0, 3]
        pose_msg.pose.position.y = self.current_pose[1, 3]
        pose_msg.pose.position.z = self.current_pose[2, 3]
        
        # Извлечение ориентации
        rotation = R.from_matrix(self.current_pose[:3, :3])
        quat = rotation.as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)

    def publish_tf(self, stamp):
        """Публикация преобразования TF"""
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'ndt_pose'
        
        # Позиция
        transform.transform.translation.x = self.current_pose[0, 3]
        transform.transform.translation.y = self.current_pose[1, 3]
        transform.transform.translation.z = self.current_pose[2, 3]
        
        # Ориентация
        rotation = R.from_matrix(self.current_pose[:3, :3])
        quat = rotation.as_quat()
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(transform)

    def publish_map(self):
        """Публикация карты как PointCloud2"""
        if not self.map_initialized or len(self.global_map_points) == 0:
            return
        
        try:
            with self.lock:
                current_map_points = self.global_map_points.copy()
            
            if len(current_map_points) == 0:
                return
            
            # Создание сообщения PointCloud2 с правильным заголовком
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'map'
            
            # Создание PointCloud2
            pc2_msg = pc2.create_cloud_xyz32(header, current_map_points.tolist())
            self.map_pub.publish(pc2_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing map: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = NdtSlamNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
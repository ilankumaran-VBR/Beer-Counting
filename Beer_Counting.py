import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from datetime import datetime, time as dt_time
import threading
import pyodbc
import uuid
from queue import Queue, Empty
import random
import os


class FrameCaptureManager:

    def __init__(self, save_folder="captured_frames", buffer_size=6):
        self.save_folder = save_folder
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            print(f"[CAPTURE] Created frame save folder: {self.save_folder}")

    def add_frame_to_buffer(self, frame):
        with self.buffer_lock:
            # Store a copy of the frame
            self.frame_buffer.append(frame.copy())

    def save_buffered_frames(self, total_count):
        with self.buffer_lock:
            if len(self.frame_buffer) == 0:
                print("[CAPTURE] No frames in buffer to save")
                return

            current_date = datetime.now().strftime("%d")
            random_num = random.randint(1000, 9999)
            base_filename = f"{total_count}_{current_date}_{random_num}"

            saved_count = 0
            for idx, frame in enumerate(self.frame_buffer):
                filename = f"{base_filename}_frame{idx + 1}.jpg"
                filepath = os.path.join(self.save_folder, filename)

                try:
                    cv2.imwrite(filepath, frame)
                    saved_count += 1
                except Exception as e:
                    print(f"[CAPTURE] Error saving frame {idx + 1}: {e}")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[CAPTURE] [{timestamp}] Saved {saved_count}/{len(self.frame_buffer)} frames as: {base_filename}_frameX.jpg")

    def get_buffer_status(self):
        with self.buffer_lock:
            return {
                'current_size': len(self.frame_buffer),
                'max_size': self.buffer_size
            }


class DatabaseLogger:
    def __init__(self, server, database, username, password):

        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection_string = (
            f'DRIVER={{FreeTDS}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
            f'TDS_Version=7.4;'
            f'Encrypt=no;'
        )
        self.retry_queue = Queue()
        self.stop_flag = False
        self.retry_interval = 300 
        self.last_retry_attempt = time.time()
        self.successful_logs = 0
        self.failed_logs = 0
        self.pending_retries = 0
        self.retry_thread = threading.Thread(target=self._retry_worker, daemon=True)
        self.retry_thread.start()
        self._ensure_table_exists()

    def _get_connection(self):
        try:
            conn = pyodbc.connect(self.connection_string, timeout=10)
            return conn
        except Exception as e:
            print(f"[DB] Connection error: {e}")
            return None

    def _ensure_table_exists(self):
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='WetItemTransactionLog' AND xtype='U')
        CREATE TABLE [dbo].[WetItemTransactionLog](
            [Id] [int] IDENTITY(1,1) NOT NULL PRIMARY KEY,
            [ItemCount] [int] NULL,
            [log_time] [datetime] NULL,
            [uuid] [nvarchar](50) NULL,
            [TransId] [int] NULL,
            [Z_Report_Id] [nvarchar](20) NULL,
            [Shift_report_Id] [nvarchar](20) NULL,
            [Tick] [bit] NULL,
            [ISUploaded] [bit] NULL
        )
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = self._get_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute(create_table_sql)
                    conn.commit()
                    cursor.close()
                    conn.close()
                    print("[DB] ✓ Table verified/created successfully")
                    return True
            except Exception as e:
                print(f"[DB] Table creation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        print("[DB] ⚠ Could not verify table - will retry on first log")
        return False

    def log_beer_count(self, item_count):
        log_time = datetime.now()
        log_uuid = str(uuid.uuid4())

        log_data = {
            'ItemCount': item_count,
            'log_time': log_time,
            'uuid': log_uuid,
            'TransId': 0,
            'Z_Report_Id': '',
            'Shift_report_Id': '',
            'Tick': 0,
            'ISUploaded': 0
        }

        success = self._insert_log(log_data)
        if not success:
            print(f"[DB] First attempt failed, trying again immediately...")
            time.sleep(0.5)  # Brief pause
            success = self._insert_log(log_data)

        if success:
            self.successful_logs += 1
            print(f"[DB] ✓ Logged to database: Count={item_count}, UUID={log_uuid[:8]}...")
        else:
            self.failed_logs += 1
            self.retry_queue.put(log_data)
            self.pending_retries = self.retry_queue.qsize()
            print(f"[DB] ✗ Failed to log - added to retry queue (pending: {self.pending_retries})")

        return success

    def _insert_log(self, log_data):
        """
        Insert log data into database AND VERIFY it was saved
        """
        insert_sql = """
        INSERT INTO [dbo].[WetItemTransactionLog] 
        (ItemCount, log_time, uuid, TransId, Z_Report_Id, Shift_report_Id, Tick, ISUploaded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        verify_sql = """
        SELECT COUNT(*) 
        FROM [dbo].[WetItemTransactionLog] 
        WHERE uuid = ?
        """

        conn = None
        try:
            conn = self._get_connection()
            if not conn:
                print(f"[DB] Failed to connect to database")
                return False

            cursor = conn.cursor()
            cursor.execute(insert_sql, (
                log_data['ItemCount'],
                log_data['log_time'],
                log_data['uuid'],
                log_data['TransId'],
                log_data['Z_Report_Id'],
                log_data['Shift_report_Id'],
                log_data['Tick'],
                log_data['ISUploaded']
            ))
            conn.commit()
            time.sleep(0.1)
            cursor.execute(verify_sql, (log_data['uuid'],))
            result = cursor.fetchone()

            if result is None:
                print(f"[DB] ⚠ Verification query returned no results for UUID {log_data['uuid'][:8]}...")
                cursor.close()
                conn.close()
                return False

            count = result[0]
            cursor.close()
            conn.close()
            if count > 0:
                return True  
            else:
                print(
                    f"[DB] ✗ UUID verification FAILED - UUID {log_data['uuid'][:8]}... NOT found in database after insert!")
                return False

        except Exception as e:
            print(f"[DB] Insert/verification error: {e}")
            if conn:
                try:
                    conn.close()
                except:
                    pass
            return False


    def _retry_worker(self):
        print("[DB] Retry worker started (checking every 30 seconds)")

        while not self.stop_flag:
            try:
                current_time = time.time()

                if current_time - self.last_retry_attempt >= 30:  # Changed from 300 to 30
                    self.last_retry_attempt = current_time

                    if not self.retry_queue.empty():
                        queue_size = self.retry_queue.qsize()
                        print(f"[DB] Retrying {queue_size} queued logs...")

                        successful_retries = 0
                        failed_items = []

                        while not self.retry_queue.empty():
                            try:
                                log_data = self.retry_queue.get_nowait()

                                if self._insert_log(log_data):
                                    successful_retries += 1
                                    self.successful_logs += 1
                                    print(
                                        f"[DB] ✓ Retry OK: ItemCount={log_data['ItemCount']}, Time={log_data['log_time'].strftime('%H:%M:%S')}")
                                else:
                                    failed_items.append(log_data)
                            except Empty:
                                break

                        for item in failed_items:
                            self.retry_queue.put(item)

                        self.pending_retries = self.retry_queue.qsize()

                        if successful_retries > 0:
                            print(f"[DB] ✓ Retry summary: {successful_retries}/{queue_size} successful")
                        if self.pending_retries > 0:
                            print(f"[DB] ⚠ {self.pending_retries} logs still pending (will retry in 30s)")

                time.sleep(5)  

            except Exception as e:
                print(f"[DB] Retry worker error: {e}")
                time.sleep(10)

    def get_statistics(self):
        return {
            'successful': self.successful_logs,
            'failed': self.failed_logs,
            'pending_retries': self.pending_retries
        }

    def shutdown(self):
        print("[DB] Shutting down database logger...")
        self.stop_flag = True
        if self.retry_thread.is_alive():
            self.retry_thread.join(timeout=5)
        print(
            f"[DB] Final stats - Success: {self.successful_logs}, Failed: {self.failed_logs}, Pending: {self.pending_retries}")


class BeerGlassRTSPCounter:
    def __init__(self, model_path, rtsp_url, counting_line, db_logger=None, frame_capture_manager=None, reset_time="04:00"):
        self.model = YOLO(model_path, task='detect')
        self.rtsp_url = rtsp_url
        self.counting_lines = counting_line
        self.db_logger = db_logger
        self.frame_capture_manager = frame_capture_manager

        self.class_names = {
            0: "Beer Glass",
            1: "Water Glass",
            2: "Empty Glass",
            3: "Cocktail Glass"
        }
        self.class_colors = {
            0: (0, 255, 0),  
            1: (255, 0, 0),  
            2: (0, 165, 255),  
            3: (255, 0, 255)  
        }
        self.tracked_objects = {}
        self.next_object_id = 0
        self.beer_count = 0
        self.counted_ids = set()
        self.counted_ids_timestamps = {}
        self.counted_id_timeout = 180

        self.near_line_objects = {}  
        self.frames_to_wait = 3  
        self.near_line_distance = 50  
        self.current_frame_number = 0

        self.last_hidden_locations = []  
        self.max_hidden_history = 3  
        self.location_overlap_threshold = 0.70 
        self.reset_time = reset_time
        self.last_reset_date = datetime.now().date()
        self.reset_lock = threading.Lock()
        self.processing_width = 640
        self.target_fps = 4
        self.frame_interval = 1.0 / self.target_fps
        self.screen_width = 1920
        self.screen_height = 1020
        self.display_scale = 0.8
        self.display_width = int(self.screen_width * self.display_scale)
        self.display_height = int(self.screen_height * self.display_scale)
        self.fps_timestamps = deque(maxlen=35)
        self.actual_fps = 0.0
        self.current_frame = None
        self.frame_condition = threading.Condition()
        self.stop_flag = False
        self.connection_lost = False
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.gray_frames_dropped = 0
        self.connection_errors = 0
        self.start_time = None
        self.last_successful_frame = None
        self.last_stats_print = None
        self.gray_check_counter = 0
        self.gray_check_interval = 10
        self.last_frame_time = time.time()
        self.frame_timeout = 30
        self.reconnect_delay = 10
        self.max_reconnect_time = 300

    def is_on_pouring_side(self, centroid, lines_coords):
        cx, cy = centroid
        near_line_distance = 30
        min_distance_to_any_line = float('inf')
        closest_line_index = None

        for idx, (line_start, line_end) in enumerate(lines_coords):
            x1, y1 = line_start
            x2, y2 = line_end

            line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if line_length_sq == 0:
                continue

            t = max(0, min(1, ((cx - x1) * (x2 - x1) + (cy - y1) * (y2 - y1)) / line_length_sq))
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)

            distance = np.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2)

            if distance < min_distance_to_any_line:
                min_distance_to_any_line = distance
                closest_line_index = idx

        if closest_line_index is None:
            return False

        line_start, line_end = lines_coords[closest_line_index]
        x1, y1 = line_start
        x2, y2 = line_end

        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_length_sq == 0:
            return False

        t = max(0, min(1, ((cx - x1) * (x2 - x1) + (cy - y1) * (y2 - y1)) / line_length_sq))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        distance = np.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2)

        if distance < near_line_distance:
            cross_product = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
            if cross_product < 0:
                return True

        return False

    def calculate_bbox_overlap(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0  # No overlap

        intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        iou = intersection_area / union_area
        return iou

    def is_duplicate_location(self, bbox):
        for prev_bbox in self.last_hidden_locations:
            overlap = self.calculate_bbox_overlap(bbox, prev_bbox)
            if overlap >= self.location_overlap_threshold:
                return True
        return False

    def add_hidden_location(self, bbox):
        self.last_hidden_locations.append(bbox)

        # Keep only last 3 locations
        if len(self.last_hidden_locations) > self.max_hidden_history:
            self.last_hidden_locations.pop(0)

    def check_disappeared_near_line(self, current_tracked_ids, lines_processing):
        disappeared_ids = []

        for obj_id, data in list(self.near_line_objects.items()):
            if obj_id not in current_tracked_ids:
                frames_since_seen = self.current_frame_number - data['last_seen_frame']

                if frames_since_seen >= self.frames_to_wait:
                    if data['class'] == 0 and obj_id not in self.counted_ids:
                        if data['frames_seen'] < 2:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(
                                f"[{timestamp}] Beer Glass seen only {data['frames_seen']} frame(s) - NOT counted (ID: {obj_id})")
                            disappeared_ids.append(obj_id)
                            continue

                        moved_away = self.check_movement_away_from_line(
                            data.get('centroid_history', []),
                            lines_processing
                        )

                        if not moved_away:
                            last_bbox = data.get('bbox')
                            last_centroid = data.get('centroid_history', [None])[-1]

                            if last_bbox is not None and last_centroid is not None:
                                on_pouring_side = self.is_on_pouring_side(last_centroid, lines_processing)

                                if not on_pouring_side:
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    print(
                                        f"[{timestamp}] Beer Glass NOT on pouring side (above line) - NOT counted (ID: {obj_id})")
                                    disappeared_ids.append(obj_id)
                                    continue
                                is_duplicate = self.is_duplicate_location(last_bbox)

                                if is_duplicate:
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    print(
                                        f"[{timestamp}] Beer Glass at duplicate location - NOT counted (ID: {obj_id})")
                                else:
      
                                    self.beer_count += 1
                                    self.counted_ids.add(obj_id)
                                    self.counted_ids_timestamps[obj_id] = time.time()  # Store timestamp
                                    self.add_hidden_location(last_bbox)

                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    print(
                                        f"[{timestamp}] ✓ Beer Glass counted (hidden, {data['frames_seen']} frames)! Total: {self.beer_count}")

                                    if self.frame_capture_manager:
                                        self.frame_capture_manager.save_buffered_frames(self.beer_count)

                                    if self.db_logger:
                                        self.db_logger.log_beer_count(1)
                            else:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                print(
                                    f"[{timestamp}] Beer Glass disappeared but no bbox/centroid - NOT counted (ID: {obj_id})")
                        else:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[{timestamp}] Beer Glass moved away from line - NOT counted (ID: {obj_id})")

                    disappeared_ids.append(obj_id)
            else:
                self.near_line_objects[obj_id]['last_seen_frame'] = self.current_frame_number

        for obj_id in disappeared_ids:
            del self.near_line_objects[obj_id]

    def check_movement_away_from_line(self, centroid_history, lines_processing):
        if len(centroid_history) < 2:
            return False

        last_pos = centroid_history[-1]
        prev_pos = centroid_history[-2]

        last_distance = self.min_distance_to_lines(last_pos, lines_processing)
        prev_distance = self.min_distance_to_lines(prev_pos, lines_processing)

        distance_change = last_distance - prev_distance

        movement_threshold = 10  

        if distance_change > movement_threshold:
            return True  

        return False  

    def min_distance_to_lines(self, centroid, lines_coords):
        cx, cy = centroid
        min_dist = float('inf')

        for line_start, line_end in lines_coords:
            x1, y1 = line_start
            x2, y2 = line_end

            line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if line_length_sq == 0:
                dist = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
            else:
                t = max(0, min(1, ((cx - x1) * (x2 - x1) + (cy - y1) * (y2 - y1)) / line_length_sq))
                proj_x = x1 + t * (x2 - x1)
                proj_y = y1 + t * (y2 - y1)
                dist = np.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2)

            min_dist = min(min_dist, dist)

        return min_dist

    def is_gray_frame(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            mean_sat = np.mean(saturation)
            return mean_sat < 15
        except Exception as e:
            print(f"Error checking gray frame: {e}")
            return True

    def resize_for_processing(self, frame):
        try:
            orig_height, orig_width = frame.shape[:2]
            aspect_ratio = orig_height / orig_width
            target_height = int(self.processing_width * aspect_ratio)
            resized = cv2.resize(frame, (self.processing_width, target_height),
                                 interpolation=cv2.INTER_AREA)
            return resized
        except Exception as e:
            print(f"Error resizing frame: {e}")
            return None

    def resize_for_display(self, frame):
        try:
            orig_height, orig_width = frame.shape[:2]
            aspect_ratio = orig_width / orig_height

            if aspect_ratio > (self.display_width / self.display_height):
                new_width = self.display_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = self.display_height
                new_width = int(new_height * aspect_ratio)

            resized = cv2.resize(frame, (new_width, new_height),
                                 interpolation=cv2.INTER_LINEAR)
            return resized
        except Exception as e:
            print(f"Error resizing for display: {e}")
            return None

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def cross_line(self, prev_pos, curr_pos, line_start, line_end):
        if prev_pos is None:
            return False

        if curr_pos[1] <= prev_pos[1]:
            return False

        x1, y1 = prev_pos
        x2, y2 = curr_pos
        x3, y3 = line_start
        x4, y4 = line_end

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return False

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            return True

        return False

    def update_tracking(self, detections):
        current_centroids = {}
        FRAME_TO_FRAME_THRESHOLD = 30
        MIN_SEPARATION_DISTANCE = 20

        for detection in detections:
            bbox = detection['bbox']
            cls = detection['class']
            centroid = self.get_centroid(bbox)

            min_distance = float('inf')
            matched_id = None
            for obj_id, data in self.tracked_objects.items():
                if data['class'] == cls:
                    prev_centroid = data['centroid']
                    distance = np.sqrt((centroid[0] - prev_centroid[0]) ** 2 +
                                       (centroid[1] - prev_centroid[1]) ** 2)

                    if distance < min_distance and distance < FRAME_TO_FRAME_THRESHOLD:
                        min_distance = distance
                        matched_id = obj_id

            too_close_to_existing = False
            if matched_id is not None:
                for existing_id, existing_data in current_centroids.items():
                    if existing_data['class'] == cls:
                        existing_centroid = existing_data['centroid']
                        distance_to_existing = np.sqrt(
                            (centroid[0] - existing_centroid[0]) ** 2 +
                            (centroid[1] - existing_centroid[1]) ** 2
                        )
                        if distance_to_existing < MIN_SEPARATION_DISTANCE:
                            too_close_to_existing = True
                            break

            if matched_id is not None:
                centroid_history = self.tracked_objects[matched_id].get('centroid_history', [])
                centroid_history.append(centroid)
                if len(centroid_history) > 3:
                    centroid_history = centroid_history[-3:]

                current_centroids[matched_id] = {
                    'centroid': centroid,
                    'prev_centroid': self.tracked_objects[matched_id]['centroid'],
                    'bbox': bbox,
                    'class': cls,
                    'centroid_history': centroid_history
                }
            else:
                has_significant_overlap = False
                for existing_id, existing_data in current_centroids.items():
                    existing_bbox = existing_data['bbox']
                    overlap = self.calculate_bbox_overlap(bbox, existing_bbox)
                    if overlap >= 0.70:  # 70% overlap threshold
                        has_significant_overlap = True
                        break

                if not has_significant_overlap:
                    current_centroids[self.next_object_id] = {
                        'centroid': centroid,
                        'prev_centroid': None,
                        'bbox': bbox,
                        'class': cls,
                        'centroid_history': [centroid]
                    }
                    self.next_object_id += 1

        if len(self.counted_ids) > 1000:
            self.counted_ids = set(list(self.counted_ids)[-500:])

        self.tracked_objects = current_centroids
        return current_centroids

    def calculate_actual_fps(self):
        if len(self.fps_timestamps) < 2:
            return 0.0

        current_time = time.time()
        recent_timestamps = [ts for ts in self.fps_timestamps if current_time - ts <= 10.0]

        if len(recent_timestamps) < 2:
            return 0.0

        time_span = recent_timestamps[-1] - recent_timestamps[0]
        if time_span > 0:
            return (len(recent_timestamps) - 1) / time_span
        return 0.0

    def draw_counting_line(self, frame, scale_factor):
        scaled_lines = []
        for i, line in enumerate(self.counting_lines):
            line_start = (int(line[0][0] * scale_factor),
                          int(line[0][1] * scale_factor))
            line_end = (int(line[1][0] * scale_factor),
                        int(line[1][1] * scale_factor))

            cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            scaled_lines.append((line_start, line_end))
        cv2.putText(frame, "COUNT ->", (scaled_lines[0][0][0] - 60, scaled_lines[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return scaled_lines

    def cleanup_old_counted_ids(self):
        current_time = time.time()
        ids_to_remove = []

        for obj_id, timestamp in self.counted_ids_timestamps.items():
            if current_time - timestamp > self.counted_id_timeout:
                ids_to_remove.append(obj_id)

        for obj_id in ids_to_remove:
            self.counted_ids.discard(obj_id)
            del self.counted_ids_timestamps[obj_id]

        if ids_to_remove:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp_str}] Cleaned up {len(ids_to_remove)} old counted IDs (timeout: {self.counted_id_timeout}s)")

    def print_statistics(self):
        current_time = time.time()

        if self.last_stats_print is None:
            self.last_stats_print = current_time
            return

        if current_time - self.last_stats_print >= 600:
            elapsed_time = int(current_time - self.start_time)
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60

            print(f"\n{'=' * 60}")
            print(f"[STATS] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 60}")
            print(f"Runtime:              {hours:02d}h {minutes:02d}m")
            print(f"Beer Count:           {self.beer_count}")
            print(f"Frames Processed:     {self.total_frames_processed}")
            print(f"Frames Dropped:       {self.total_frames_dropped}")
            print(f"Gray Frames:          {self.gray_frames_dropped}")
            print(f"Connection Errors:    {self.connection_errors}")
            print(f"Current FPS:          {self.actual_fps:.2f}")

            if self.db_logger:
                db_stats = self.db_logger.get_statistics()
                print(f"DB Logs Success:      {db_stats['successful']}")
                print(f"DB Logs Failed:       {db_stats['failed']}")
                print(f"DB Pending Retries:   {db_stats['pending_retries']}")

            print(f"{'=' * 60}\n")

            self.last_stats_print = current_time

    def connect_rtsp(self):
        print(f"Connecting to RTSP stream: {self.rtsp_url}")
        cap = cv2.VideoCapture(self.rtsp_url)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 400)
        cap.set(cv2.CAP_PROP_FPS, 15)

        if not cap.isOpened():
            raise ConnectionError(f"Failed to connect to RTSP stream: {self.rtsp_url}")

        print("✓ Successfully connected to RTSP stream")
        self.connection_lost = False
        return cap

    def frame_grabber_thread(self, cap):
        consecutive_failures = 0
        max_consecutive_failures = 30

        while not self.stop_flag:
            try:
                ret, frame = cap.read()
                if ret:
                    with self.frame_condition:
                        self.current_frame = frame
                        self.last_frame_time = time.time()
                        self.frame_condition.notify()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"Frame grabber: {consecutive_failures} consecutive failures")
                        self.connection_lost = True
                        break
                    time.sleep(0.1)
            except Exception as e:
                print(f"Frame grabber error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    self.connection_lost = True
                    break
                time.sleep(0.1)

    def check_watchdog(self):
        current_time = time.time()
        if current_time - self.last_frame_time > self.frame_timeout:
            print("Watchdog: No frames received for 30 seconds")
            return False
        return True

    def check_and_perform_reset(self):

        with self.reset_lock:
            now = datetime.now()
            current_date = now.date()
            current_time = now.time()


            reset_hour, reset_minute = map(int, self.reset_time.split(':'))
            reset_time_obj = dt_time(reset_hour, reset_minute)


            if current_date > self.last_reset_date:

                if current_time >= reset_time_obj:
                    self.perform_reset()
                    self.last_reset_date = current_date
            elif current_date == self.last_reset_date:

                time_since_midnight = current_time.hour * 60 + current_time.minute
                reset_minutes = reset_hour * 60 + reset_minute


                if abs(time_since_midnight - reset_minutes) <= 1:

                    if hasattr(self, 'last_reset_timestamp'):
                        if (time.time() - self.last_reset_timestamp) > 82800:  
                            self.perform_reset()
                            self.last_reset_date = current_date
                    else:
                        self.perform_reset()
                        self.last_reset_date = current_date

    def perform_reset(self):
        old_count = self.beer_count
        self.beer_count = 0
        self.counted_ids.clear()
        self.counted_ids_timestamps.clear()
        self.tracked_objects = {}
        self.near_line_objects = {}
        self.current_frame_number = 0
        self.next_object_id = 0
        self.last_hidden_locations = []
        self.last_reset_timestamp = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 60}")
        print(f"[DAILY RESET] {timestamp}")
        print(f"{'=' * 60}")
        print(f"Previous count: {old_count}")
        print(f"Count reset to: 0")
        print(f"All tracking data cleared")
        print(f"Hidden location history cleared")
        print(f"Next reset scheduled for: {self.reset_time}")
        print(f"{'=' * 60}\n")

    def process_stream(self):
        cap = None
        grabber_thread = None

        cv2.namedWindow('Beer Counter - RPi', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Beer Counter - RPi', self.display_width, self.display_height)

        window_x = (self.screen_width - self.display_width) // 2
        window_y = (self.screen_height - self.display_height) // 2
        cv2.moveWindow('Beer Counter - RPi', window_x, window_y)

        while True:
            try:
                if cap is None or self.connection_lost:
                    if cap is not None:
                        cap.release()

                    reconnect_start_time = time.time()
                    reconnect_attempt = 1

                    while True:
                        try:
                            elapsed_reconnect = time.time() - reconnect_start_time

                            if elapsed_reconnect > self.max_reconnect_time:
                                delay = 60
                            else:
                                delay = self.reconnect_delay

                            if reconnect_attempt > 1:
                                print(f"Reconnection attempt #{reconnect_attempt} "
                                      f"(elapsed: {int(elapsed_reconnect)}s, waiting {delay}s...)")
                                time.sleep(delay)

                            cap = self.connect_rtsp()
                            break

                        except Exception as e:
                            print(f"Connection failed: {e}")
                            reconnect_attempt += 1

                            if self.stop_flag:
                                return

                    self.connection_lost = False
                    self.connection_errors += 1

                    if grabber_thread is not None:
                        self.stop_flag = True
                        grabber_thread.join(timeout=2.0)
                        self.stop_flag = False

                    grabber_thread = threading.Thread(target=self.frame_grabber_thread,
                                                      args=(cap,), daemon=True)
                    grabber_thread.start()

                    if self.start_time is None:
                        self.start_time = time.time()
                        self.last_stats_print = time.time()

                print(f"Processing at: {self.processing_width}px width")
                print(f"Display at: 80% screen size ({self.display_width}x{self.display_height} max)")
                print(f"Target processing rate: {self.target_fps} FPS")
                if self.db_logger:
                    print("✓ Database logging ENABLED")
                print("Press 'q' to quit")
                print("Press 'r' to reset count")
                print("Stats will print every 10 minutes\n")

                last_process_time = 0
                last_watchdog_check = time.time()

                while not self.connection_lost:
                    current_time = time.time()

                    if current_time - last_watchdog_check > 5.0:
                        if not self.check_watchdog():
                            self.connection_lost = True
                            print("Connection lost. Will attempt to reconnect...")
                            break
                        last_watchdog_check = current_time

                    self.print_statistics()
                    self.cleanup_old_counted_ids()

                    if current_time - last_process_time < self.frame_interval:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.stop_flag = True
                            return
                        elif key == ord('r'):
                            self.beer_count = 0
                            self.counted_ids.clear()
                            print("Count reset to 0")
                        continue

                    with self.frame_condition:
                        if self.current_frame is None:
                            continue
                        frame = self.current_frame.copy()

                    last_process_time = current_time
                    self.current_frame_number += 1

                    self.gray_check_counter += 1
                    if self.gray_check_counter >= self.gray_check_interval:
                        if self.is_gray_frame(frame):
                            self.gray_frames_dropped += 1
                            self.total_frames_dropped += 1
                            self.gray_check_counter = 0
                            continue
                        self.gray_check_counter = 0

                    self.total_frames_processed += 1
                    self.fps_timestamps.append(current_time)

                    processing_frame = self.resize_for_processing(frame)
                    if processing_frame is None:
                        continue

                    if self.frame_capture_manager:
                        self.frame_capture_manager.add_frame_to_buffer(processing_frame)

                    results = self.model(processing_frame, verbose=False, imgsz=640)

                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            bbox = boxes.xyxy[i].cpu().numpy()
                            cls = int(boxes.cls[i])
                            conf = float(boxes.conf[i])

                            if conf > 0.5:
                                detections.append({
                                    'bbox': bbox,
                                    'class': cls,
                                    'confidence': conf
                                })

                    tracked = self.update_tracking(detections)

                    processing_scale = self.processing_width / 1280

                    lines_processing = []
                    for line in self.counting_lines:
                        line_start = (int(line[0][0] * processing_scale),
                                      int(line[0][1] * processing_scale))
                        line_end = (int(line[1][0] * processing_scale),
                                    int(line[1][1] * processing_scale))
                        lines_processing.append((line_start, line_end))

                    for obj_id, data in tracked.items():
                        cls = data['class']
                        centroid = data['centroid']
                        prev_centroid = data['prev_centroid']

                        # Check if Beer Glass crossed ANY of the 3 lines (right to left)
                        if cls == 0 and obj_id not in self.counted_ids:
                            crossed = False
                            for line_seg in lines_processing:
                                if self.cross_line(prev_centroid, centroid, line_seg[0], line_seg[1]):
                                    crossed = True
                                    break

                            if crossed:
                                self.beer_count += 1
                                self.counted_ids.add(obj_id)
                                self.counted_ids_timestamps[obj_id] = time.time()  # Store timestamp
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                print(f"[{timestamp}] Beer Glass counted! Total: {self.beer_count}")

                                if self.frame_capture_manager:
                                    self.frame_capture_manager.save_buffered_frames(self.beer_count)

                                # Log to database
                                if self.db_logger:
                                    self.db_logger.log_beer_count(1)

                    for obj_id, data in tracked.items():
                        cls = data['class']
                        centroid = data['centroid']
                        bbox = data['bbox']

                        if cls == 0 and obj_id not in self.counted_ids:
                            if self.is_on_pouring_side(centroid, lines_processing):
                                if obj_id not in self.near_line_objects:
                                    self.near_line_objects[obj_id] = {
                                        'frames_seen': 1,
                                        'last_seen_frame': self.current_frame_number,
                                        'class': cls,
                                        'centroid_history': [centroid],
                                        'bbox': bbox 
                                    }
                                else:
                                    self.near_line_objects[obj_id]['frames_seen'] += 1
                                    self.near_line_objects[obj_id]['last_seen_frame'] = self.current_frame_number
                                    self.near_line_objects[obj_id]['bbox'] = bbox  

                                    if 'centroid_history' not in self.near_line_objects[obj_id]:
                                        self.near_line_objects[obj_id]['centroid_history'] = []

                                    self.near_line_objects[obj_id]['centroid_history'].append(centroid)


                                    if len(self.near_line_objects[obj_id]['centroid_history']) > 3:
                                        self.near_line_objects[obj_id]['centroid_history'].pop(0)
                            else:

                                if obj_id in self.near_line_objects:
                                    del self.near_line_objects[obj_id]

                    current_tracked_ids = set(tracked.keys())
                    self.check_disappeared_near_line(current_tracked_ids, lines_processing)

                    display_frame = self.resize_for_display(processing_frame)
                    if display_frame is None:
                        continue

                    display_scale = display_frame.shape[1] / self.processing_width

                    # Draw counting line
                    lines_display = self.draw_counting_line(display_frame, processing_scale * display_scale)

                    for obj_id, data in tracked.items():
                        bbox = data['bbox']
                        cls = data['class']

                        x1, y1, x2, y2 = bbox * display_scale
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        color = self.class_colors.get(cls, (255, 255, 255))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                        class_name = self.class_names.get(cls, f"Class {cls}")
                        cv2.putText(display_frame, class_name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    self.actual_fps = self.calculate_actual_fps()

                    panel_width = 180
                    panel_height = 120
                    frame_height, frame_width = display_frame.shape[:2]

                    panel_x = frame_width - panel_width - 10
                    panel_y = 10

                    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
                    panel[:] = (40, 40, 40)

                    alpha = 0.85
                    display_frame[panel_y:panel_y + panel_height,
                    panel_x:panel_x + panel_width] = cv2.addWeighted(
                        display_frame[panel_y:panel_y + panel_height,
                        panel_x:panel_x + panel_width], 1 - alpha,
                        panel, alpha, 0
                    )

                    cv2.rectangle(display_frame, (panel_x, panel_y),
                                  (panel_x + panel_width, panel_y + panel_height),
                                  (80, 80, 80), 2)

                    cv2.putText(display_frame, "BEER COUNT", (panel_x + 15, panel_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.line(display_frame, (panel_x + 15, panel_y + 38),
                             (panel_x + panel_width - 15, panel_y + 38), (255, 255, 255), 1)

                    cv2.putText(display_frame, str(self.beer_count), (panel_x + 15, panel_y + 80),
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

                    fps_color = (0, 255, 0) if self.actual_fps >= 3.0 else (0, 165, 255)
                    cv2.putText(display_frame, f"FPS: {self.actual_fps:.1f}",
                                (panel_x + 15, panel_y + 108),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)

                    cv2.imshow('Beer Counter - RPi', display_frame)
                    self.check_and_perform_reset()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_flag = True
                        return
                    elif key == ord('r'):
                        self.beer_count = 0
                        self.counted_ids.clear()
                        print("Count reset to 0")

            except KeyboardInterrupt:
                print("\nShutting down...")
                self.stop_flag = True
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                self.connection_lost = True
                time.sleep(5)

        self.stop_flag = True
        if grabber_thread:
            grabber_thread.join(timeout=2.0)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        print(f"\n=== Final Session Summary ===")
        print(f"Total Beer Glasses Counted: {self.beer_count}")
        print(f"Total Frames Processed: {self.total_frames_processed}")
        print(f"Total Frames Dropped: {self.total_frames_dropped}")
        print(f"Gray Frames Dropped: {self.gray_frames_dropped}")
        print(f"Connection Errors: {self.connection_errors}")
        print(f"Final Average FPS: {self.actual_fps:.2f}")
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"Total Runtime: {int(total_time)} seconds ({total_time / 3600:.1f} hours)")

        if self.db_logger:
            db_stats = self.db_logger.get_statistics()
            print(f"\n=== Database Summary ===")
            print(f"Successful Logs: {db_stats['successful']}")
            print(f"Failed Logs: {db_stats['failed']}")
            print(f"Pending Retries: {db_stats['pending_retries']}")


if __name__ == "__main__":

    MODEL_PATH = "model"
    RTSP_URL = "rtsp://admin:xxxxx@xx.xxx.xxx.xxxx:554/Streaming/Channels/101"
    COUNTING_LINES = [
        ((99, 393), (127, 491)),
        ((127, 491), (337, 538)),
        ((337, 538), (853, 571)),
        ((853, 571), (1236, 546))
    ]
    DB_SERVER = "xx.xx.xx.xxx,xxx"  
    DB_NAME = "xxx"
    DB_USERNAME = "xx"
    DB_PASSWORD = "xxxxx"
    FRAME_SAVE_FOLDER = "captured_frames"  
    FRAME_BUFFER_SIZE = 6

    try:
        print("Initializing database logger...")
        db_logger = DatabaseLogger(DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD)
        print("✓ Database logger initialized\n")

        print("Initializing frame capture manager...")
        frame_capture_manager = FrameCaptureManager(
            save_folder=FRAME_SAVE_FOLDER,
            buffer_size=FRAME_BUFFER_SIZE
        )
        print("✓ Frame capture manager initialized\n")
        counter = BeerGlassRTSPCounter(MODEL_PATH, RTSP_URL, COUNTING_LINES, db_logger, frame_capture_manager)
        counter.process_stream()

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if 'db_logger' in locals():
            db_logger.shutdown()
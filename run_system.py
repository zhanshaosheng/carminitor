import sys
import cv2  # 计算机视觉库，用于视频处理和图像操作
import numpy as np  # 数值计算库，用于处理矩阵和数组
import pandas as pd  # 数据处理库，用于处理检测结果
import time  # 时间库，用于计算违停时间
import os
import platform
import subprocess

# 跨平台 TTS 语音播报
def _cross_platform_speak(text):
    """跨平台语音播报：macOS 用 say，Windows 回退蜂鸣，Linux 用 espeak。"""
    system = platform.system()
    if system == 'Darwin':  # macOS — 内置 say 命令，后台异步不阻塞
        try:
            subprocess.Popen(['say', text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    elif system == 'Windows':
        try:
            import winsound
            winsound.Beep(1000, 400)
        except Exception:
            pass
    else:  # Linux / 其他
        try:
            subprocess.Popen(['espeak', text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
from datetime import datetime
from ultralytics import YOLO  # 导入YOLOv8模型
from tracker import *  # 导入自定义的目标跟踪模块

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                             QListWidget, QGroupBox, QSplitter, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# 加载检测模型核心
model = YOLO('yolov8n.pt')

try:
    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")
except FileNotFoundError:
    class_list = ["car"]


class ParkingViolationConfig:
    def __init__(self):
        self.violation_threshold = 10  # 默认计时防线（秒）
        self.violation_vehicles = {}  # 容器映射表： vehicle_id -> 首次压线时间戳
        self.display_time = True
        self.alert_enabled = True



class SoundAlertManager:
    """语音周期播报管理器：车辆达到违停阈值后周期性语音播报，驶离后自动停止。"""
    def __init__(self):
        self.last_announce_time = {}   # vehicle_id -> 上次播报时间戳
        self.announce_interval = 5     # 同一辆车每 5 秒最多播报一次
        self.active_violators = set()  # 当前正在周期播报的车辆 ID 集合

    def announce_violation(self, vehicle_id, violation_time):
        """达到阈值的车辆周期语音播报，内置冷却控制。"""
        current_time = time.time()
        if (vehicle_id not in self.last_announce_time or
                current_time - self.last_announce_time[vehicle_id] >= self.announce_interval):
            self.last_announce_time[vehicle_id] = current_time
            self.active_violators.add(vehicle_id)
            text = f"车辆{vehicle_id}违停{violation_time}秒"
            _cross_platform_speak(text)

    def clear_vehicle(self, vehicle_id):
        """车辆驶离违停区域时清除播报记录。"""
        self.last_announce_time.pop(vehicle_id, None)
        self.active_violators.discard(vehicle_id)

    def clear_all(self):
        """切换视频或重置区域时清除全部播报状态。"""
        self.last_announce_time.clear()
        self.active_violators.clear()


# 实例化全局免文件声音警报器
alert_player = SoundAlertManager()


# 违停检测核心类
class ParkingViolationDetector:
    def __init__(self, config):
        self.config = config

    def check_violation(self, vehicle_id, position, current_time, area_points):
        if len(area_points) != 4:
            return False
        try:
            center_x, center_y = position
            local_polygon = np.array(area_points, np.int32)
            is_in_violation_area = cv2.pointPolygonTest(local_polygon, (center_x, center_y), False) >= 0

            if is_in_violation_area:
                if vehicle_id not in self.config.violation_vehicles:
                    self.config.violation_vehicles[vehicle_id] = current_time
                return True
            else:
                if vehicle_id in self.config.violation_vehicles:
                    del self.config.violation_vehicles[vehicle_id]
                return False
        except Exception:
            return False

    def get_violation_time(self, vehicle_id, current_time):
        if vehicle_id in self.config.violation_vehicles:
            return int(current_time - self.config.violation_vehicles[vehicle_id])
        return 0

    def draw_violation_info(self, frame, x1, y1, x2, y2, vehicle_id, current_time):
        violation_time = self.get_violation_time(vehicle_id, current_time)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        info_text = f"ID: {vehicle_id}"
        if self.config.display_time:
            info_text += f" Time: {violation_time}s"
        cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if violation_time >= self.config.violation_threshold:
            cv2.putText(frame, "VIOLATION!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



class VideoThread(QThread):
    img_render_signal = pyqtSignal(np.ndarray)
    status_report_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.video_path = 'v.mp4'  # 默认打开的第一个视频
        self._run_flag = True
        self._pause_flag = False
        self.area = []

        self.parking_config = ParkingViolationConfig()
        self.violation_detector = ParkingViolationDetector(self.parking_config)
        self.tracker = Tracker()
        self.current_frame = None

    def set_video_path(self, path):
        self.video_path = path
        # 彻底重置追踪器、时间字典与语音播报状态，防止第二个视频不计时不报警
        self.parking_config.violation_vehicles.clear()
        self.tracker = Tracker()
        alert_player.clear_all()

    def toggle_pause(self):
        self._pause_flag = not self._pause_flag
        return self._pause_flag

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        frame_delay = 1.0 / fps
        count = 0

        while self._run_flag:
            if self._pause_flag:
                time.sleep(0.1)
                continue

            start_processing_time = time.time()

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.parking_config.violation_vehicles.clear()
                continue

            count += 1
            if count % 3 != 0:
                elapsed = time.time() - start_processing_time
                delay = frame_delay - elapsed
                if delay > 0:
                    time.sleep(delay)
                continue

            frame = cv2.resize(frame, (1020, 500))
            current_time = time.time()

            results = model.predict(frame, verbose=False)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            car_list = []

            for index, row in px.iterrows():
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                d = int(row[5])
                c = class_list[d] if d < len(class_list) else 'car'
                if 'car' in c:
                    car_list.append([x1, y1, x2, y2])

            bbox_id = self.tracker.update(car_list)
            current_violators = []
            current_threshold_ids = set()  # 本帧达到违停阈值的车辆 ID

            current_points = list(self.area)
            points_len = len(current_points)

            for item in bbox_id:
                x1, y1, x2, y2, id_ = item
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if points_len == 4:
                    is_violating = self.violation_detector.check_violation(id_, (center_x, center_y), current_time,
                                                                           current_points)

                    if is_violating:
                        self.violation_detector.draw_violation_info(frame, x1, y1, x2, y2, id_, current_time)
                        v_time = self.violation_detector.get_violation_time(id_, current_time)

                        if v_time >= self.parking_config.violation_threshold:
                            current_violators.append(f"车辆 ID: {id_} (违停停留 {v_time} 秒)")
                            current_threshold_ids.add(id_)
                            if self.parking_config.alert_enabled:
                                # 周期语音播报违停信息
                                alert_player.announce_violation(id_, v_time)
                        else:
                            current_violators.append(f"监控区内 ID: {id_} (已停留 {v_time} 秒)")

            # 清理已驶离的车辆：之前在播报但现在不在本帧违规集合中
            for vid in list(alert_player.active_violators):
                if vid not in current_threshold_ids:
                    alert_player.clear_vehicle(vid)

            self.status_report_signal.emit(current_violators)

            if points_len == 4:
                cv2.polylines(frame, [np.array(current_points, np.int32)], True, (255, 0, 0), 2)
            elif points_len > 0:
                cv2.polylines(frame, [np.array(current_points, np.int32)], False, (255, 0, 0), 2)

            violation_count = sum(1 for id_ in self.parking_config.violation_vehicles if
                                  self.violation_detector.get_violation_time(id_,
                                                                             current_time) >= self.parking_config.violation_threshold)
            cv2.putText(frame, f"Violations: {violation_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.current_frame = frame.copy()
            self.img_render_signal.emit(frame)

            elapsed_time = time.time() - start_processing_time
            delay = frame_delay - elapsed_time
            if delay > 0:
                time.sleep(delay)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# ==================== 3. PyQt5 图形主界面层 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车辆违章停车检测程序 v1.6")
        self.resize(1320, 680)

        self.setStyleSheet("""
            QMainWindow { background-color: #E8F0E8; }
            QLabel { color: #2E3A2E; font-family: 'Microsoft YaHei', sans-serif; font-size: 13px; }
            QPushButton { 
                background-color: #FFFFFF; color: #3B4A3B; 
                border: 1px solid #B8C6B8; border-radius: 3px; padding: 6px 12px;
                font-size: 13px; font-family: 'Microsoft YaHei';
            }
            QPushButton:hover { background-color: #F4F7F4; border-color: #8AA08A; color: #1C241C; }
            QPushButton:pressed { background-color: #D8E4D8; }
            QListWidget { 
                background-color: #FFFFFF; border: 1px solid #B8C6B8; 
                border-radius: 4px; color: #2E3A2E; font-size: 13px; padding: 5px;
            }
            QLineEdit {
                background-color: #FFFFFF; border: 1px solid #B8C6B8;
                border-radius: 3px; padding: 4px; color: #2E3A2E;
            }
            QGroupBox {
                border: 1px solid #B8C6B8; border-radius: 4px;
                margin-top: 12px; font-weight: bold; color: #2E3A2E;
                font-family: 'Microsoft YaHei';
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.top_bar = QHBoxLayout()
        self.status_label = QLabel("程序状态：正常运行")
        self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        self.info_label = QLabel("提示：在视频画面上使用鼠标左键点击 4 个点绘制检测区域")
        self.info_label.setStyleSheet("color: #617361;")
        self.top_bar.addWidget(self.status_label)
        self.top_bar.addStretch()
        self.top_bar.addWidget(self.info_label)
        self.main_layout.addLayout(self.top_bar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #CBD6CB; }")

        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1020, 500)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #B8C6B8;")
        self.left_layout.addWidget(self.video_label)
        self.splitter.addWidget(self.left_widget)

        self.right_widget = QGroupBox("当前车辆状态列表")
        self.right_layout = QVBoxLayout(self.right_widget)

        self.thresh_layout = QHBoxLayout()
        self.thresh_label = QLabel("违停判定阈值(秒):")
        self.thresh_input = QLineEdit()
        self.thresh_input.setText("10")
        self.thresh_input.setFixedWidth(50)
        self.btn_set_thresh = QPushButton("应用")
        self.thresh_layout.addWidget(self.thresh_label)
        self.thresh_layout.addWidget(self.thresh_input)
        self.thresh_layout.addWidget(self.btn_set_thresh)
        self.right_layout.addLayout(self.thresh_layout)

        self.log_list = QListWidget()
        self.right_layout.addWidget(self.log_list)

        self.btn_clear_list = QPushButton("重置/清空车辆记录")
        self.right_layout.addWidget(self.btn_clear_list)

        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([1020, 260])
        self.main_layout.addWidget(self.splitter)

        self.control_group = QGroupBox("控制台")
        self.button_layout = QHBoxLayout(self.control_group)

        self.btn_select_video = QPushButton("选择视频文件")
        self.btn_pause_video = QPushButton("暂停监控")
        self.btn_screenshot = QPushButton("保存当前帧图片")
        self.btn_clear_area = QPushButton("清除检测区域")

        self.btn_select_video.setStyleSheet("background-color: #2E7D32; color: #FFFFFF; border: none;")
        self.btn_clear_area.setStyleSheet("color: #C62828; border-color: #E57373;")

        self.button_layout.addWidget(self.btn_select_video)
        self.button_layout.addWidget(self.btn_pause_video)
        self.button_layout.addWidget(self.btn_screenshot)
        self.button_layout.addWidget(self.btn_clear_area)
        self.main_layout.addWidget(self.control_group)

        self.btn_select_video.clicked.connect(self.change_video)
        self.btn_pause_video.clicked.connect(self.play_pause_video)
        self.btn_clear_area.clicked.connect(self.clear_area)
        self.btn_screenshot.clicked.connect(self.capture_snapshot)
        self.btn_set_thresh.clicked.connect(self.update_threshold)
        self.btn_clear_list.clicked.connect(self.clear_vehicle_records)

        self.video_label.mousePressEvent = self.capture_mouse_click

        self.thread = VideoThread()
        self.thread.img_render_signal.connect(self.process_gui_frame)
        self.thread.status_report_signal.connect(self.refresh_log_panel)
        self.thread.start()

    def process_gui_frame(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    def refresh_log_panel(self, violators_list):
        self.log_list.clear()
        if not violators_list:
            self.log_list.addItem("暂无检测目标信息")
        else:
            self.log_list.addItems(violators_list)

    def update_threshold(self):
        try:
            val = int(self.thresh_input.text())
            if val > 0:
                self.thread.parking_config.violation_threshold = val
                self.info_label.setText(f"提示：成功将违停阈值修改为 {val} 秒。")
            else:
                self.info_label.setText("提示：请输入大于 0 的整数秒。")
        except ValueError:
            self.info_label.setText("提示：输入不合法，请输入整数。")

    def clear_vehicle_records(self):
        self.thread.parking_config.violation_vehicles.clear()
        self.info_label.setText("提示：当前字典内的车辆停留记录已全部重置。")

    def capture_snapshot(self):
        if self.thread.current_frame is not None:
            os.makedirs("snapshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.thread.current_frame)
            self.info_label.setText(f"提示：图片已保存至 {filename}")
        else:
            self.info_label.setText("提示：无可用画面。")

    def capture_mouse_click(self, event):
        if event.button() == Qt.LeftButton:
            click_x = event.x()
            click_y = event.y()

            if len(self.thread.area) >= 4:
                return

            self.thread.area.append([click_x, click_y])
            self.info_label.setText(f"已选取边界点: ({click_x}, {click_y})")

            if len(self.thread.area) == 4:
                self.info_label.setText("提示：检测区域设置完成，警报器就绪。")

    def clear_area(self):
        self.thread.area.clear()
        self.thread.parking_config.violation_vehicles.clear()
        self.info_label.setText("区域已清除，请重新点击 4 个点配置区域。")

    def play_pause_video(self):
        is_paused = self.thread.toggle_pause()
        if is_paused:
            self.btn_pause_video.setText("继续监控")
            self.status_label.setText("程序状态：暂停中")
            self.status_label.setStyleSheet("color: #E65100; font-weight: bold;")
        else:
            self.btn_pause_video.setText("暂停监控")
            self.status_label.setText("程序状态：正常运行")
            self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold;")

    def change_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.thread.stop()
            self.thread.set_video_path(file_name)
            self.thread.area.clear()

            self.thread._pause_flag = False
            self.btn_pause_video.setText("暂停监控")
            self.status_label.setText("程序状态：正常运行")
            self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
            self.info_label.setText("新监控载入成功。请重新在画面上框选 4 个点。")

            self.thread._run_flag = True
            self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
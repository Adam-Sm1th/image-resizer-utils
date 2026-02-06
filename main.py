import sys
import cv2
import numpy as np
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtCore import Qt


class ImageResizer(QWidget):
    def __init__(self):
        super().__init__()
        self.points = []
        self.raw_image = None
        self.processed_image = None
        self.current_display_base = None
        self.save_count = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('透视变换校正工具 (自适应尺寸版)')
        self.setMinimumSize(900, 700)
        main_layout = QVBoxLayout()

        # --- 第一行：操作按钮 ---
        row1 = QHBoxLayout()
        self.btn_open = QPushButton('1. 打开图片')
        self.btn_open.clicked.connect(self.open_image)

        self.btn_reset = QPushButton('2. 重置选点')
        self.btn_reset.clicked.connect(self.start_selecting)

        row1.addWidget(self.btn_open)
        row1.addWidget(self.btn_reset)
        row1.addStretch()
        main_layout.addLayout(row1)

        # --- 第二行：图片显示区域 ---
        self.img_label = QLabel('操作提示：点击图片选点（左上、右上、左下、右下）')
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("border: 2px solid #555; background-color: #222; border-radius: 4px; color: #aaa;")
        self.img_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.img_label.mousePressEvent = self.get_pixel_pos
        main_layout.addWidget(self.img_label, stretch=1)

        # --- 第三行：执行与保存 ---
        row3 = QHBoxLayout()
        self.btn_confirm = QPushButton('3. 执行变换 (自动计算尺寸)')
        self.btn_confirm.setMinimumHeight(50)
        self.btn_confirm.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        self.btn_confirm.clicked.connect(self.process_transform)

        self.btn_save = QPushButton('4. 保存结果')
        self.btn_save.setMinimumHeight(50)
        self.btn_save.clicked.connect(self.save_image)

        row3.addWidget(self.btn_confirm)
        row3.addWidget(self.btn_save)
        main_layout.addLayout(row3)

        self.setLayout(main_layout)

    # ... [保持 open_image, update_image_display, resize_event, get_pixel_pos 代码不变] ...

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self.current_display_base is not None:
            self.update_image_display(self.current_display_base)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Images (*.jpg *.png *.jpeg *.bmp)')
        if fname:
            self.raw_image = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.raw_image is None:
                QMessageBox.critical(self, "错误", "图片加载失败！")
                return
            self.points = []
            self.processed_image = None
            self.current_display_base = self.raw_image.copy()
            self.update_image_display(self.current_display_base)

    def update_image_display(self, cv_img):
        if cv_img is None: return
        label_w, label_h = self.img_label.width(), self.img_label.height()
        if label_w < 10 or label_h < 10: return
        h, w = cv_img.shape[:2]
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(cv_img, (new_w, new_h))
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_image.data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888).copy()
        self.img_label.setPixmap(QPixmap.fromImage(q_img))

    def start_selecting(self):
        if self.raw_image is not None:
            self.points = []
            self.processed_image = None
            self.current_display_base = self.raw_image.copy()
            self.update_image_display(self.current_display_base)

    def get_pixel_pos(self, event):
        if self.raw_image is None or len(self.points) >= 4: return
        pixmap = self.img_label.pixmap()
        if not pixmap: return
        pw, ph = pixmap.width(), pixmap.height()
        offset_x = (self.img_label.width() - pw) / 2
        offset_y = (self.img_label.height() - ph) / 2
        x, y = event.pos().x(), event.pos().y()
        if offset_x <= x <= offset_x + pw and offset_y <= y <= offset_y + ph:
            real_x = (x - offset_x) * (self.raw_image.shape[1] / pw)
            real_y = (y - offset_y) * (self.raw_image.shape[0] / ph)
            self.points.append([real_x, real_y])
            img_h, img_w = self.raw_image.shape[:2]
            base_size = np.sqrt(img_h ** 2 + img_w ** 2)
            dynamic_radius = max(3, int(base_size * 0.005))
            temp_img = self.raw_image.copy()
            for i, p in enumerate(self.points):
                cv2.circle(temp_img, (int(p[0]), int(p[1])), dynamic_radius, (0, 255, 0), -1)
                cv2.putText(temp_img, str(i + 1), (int(p[0]) + 15, int(p[1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, base_size * 0.001, (0, 0, 255), max(1, int(base_size * 0.002)))
            self.current_display_base = temp_img
            self.update_image_display(self.current_display_base)

    # --- 重点修改：自动计算尺寸的逻辑 ---
    def process_transform(self):
        if len(self.points) != 4:
            QMessageBox.warning(self, '提示', '请按顺序选满4个点（左上、右上、左下、右下）')
            return

        # 将点解包以便计算
        # pts1 的顺序假设：0:左上, 1:右上, 2:左下, 3:右下
        (tl, tr, bl, br) = self.points

        # 计算底边和顶边的宽度，取最大值
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # 计算左边和右边的高度，取最大值
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # 设置目标点
        pts1 = np.float32([tl, tr, bl, br])
        pts2 = np.float32([
            [0, 0],
            [max_width - 1, 0],
            [0, max_height - 1],
            [max_width - 1, max_height - 1]
        ])

        # 执行变换
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.processed_image = cv2.warpPerspective(self.raw_image, matrix, (max_width, max_height))

        self.current_display_base = self.processed_image.copy()
        self.update_image_display(self.current_display_base)
        QMessageBox.information(self, '提示', f'变换完成！输出尺寸: {max_width}x{max_height}')

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, '提示', '没有可保存的变换结果')
            return
        default_name = f"{self.save_count:03d}.png"
        fname, _ = QFileDialog.getSaveFileName(self, '保存', default_name, 'PNG (*.png);;JPG (*.jpg)')
        if fname:
            ext = fname.split('.')[-1].lower()
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 0] if ext == 'png' else [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            success, encoded_img = cv2.imencode(f".{ext}", self.processed_image, params)
            if success:
                encoded_img.tofile(fname)
                QMessageBox.information(self, '成功', f'图片已保存为: {os.path.basename(fname)}')
                self.save_count += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageResizer()
    window.show()
    sys.exit(app.exec())
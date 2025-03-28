import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOcSort
import torch
from pathlib import Path
import signal
import sys
import os

# Biến toàn cục
drawing = False
line_start = None
line_end = None
arrow_start = None
arrow_end = None
setup_done = False
previous_positions = {}
out = None
frame_counter = 0

# Tạo thư mục để lưu frame nếu chưa tồn tại
output_frame_dir = "vehicle_frames"
if not os.path.exists(output_frame_dir):
    os.makedirs(output_frame_dir)

# Hàm xử lý ngắt Ctrl+C
def signal_handler(sig, frame):
    print("\nĐã ngắt chương trình bằng Ctrl+C")
    if out is not None:
        out.release()
        print("Đã lưu video output")
    cv2.destroyAllWindows()
    print(f"Total vehicles counted: {vehicle_count}")
    sys.exit(0)

# Đăng ký xử lý signal Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Hàm xử lý sự kiện chuột
def draw_line_and_arrow(event, x, y, flags, param):
    global drawing, line_start, line_end, arrow_start, arrow_end, setup_done
    if not setup_done:
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing == False and line_start is None:
                line_start = (x, y)
                drawing = True
            elif drawing == True and line_end is None:
                line_end = (x, y)
                drawing = False
            elif line_end is not None and arrow_start is None:
                arrow_start = (x, y)
                drawing = True
            elif drawing == True and arrow_end is None:
                arrow_end = (x, y)
                drawing = False
                setup_done = True

# Hàm tính giao điểm
def line_intersection(p1, p2, p3, p4):
    denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if denom == 0:
        return None
    t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom
    u = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    return None

# Hàm lưu tọa độ vào file txt
def save_line_and_arrow_to_txt():
    with open("line_and_arrow.txt", "w") as f:
        f.write(f"Line Start: {line_start[0]},{line_start[1]}\n")
        f.write(f"Line End: {line_end[0]},{line_end[1]}\n")
        f.write(f"Arrow Start: {arrow_start[0]},{arrow_start[1]}\n")
        f.write(f"Arrow End: {arrow_end[0]},{arrow_end[1]}\n")
    print("Đã lưu tọa độ line và arrow vào line_and_arrow.txt")

# Khởi tạo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = YOLO("vehicles_yolov8n.pt")
tracker = DeepOcSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device=device,
    half=True,
    det_thresh=0.5,
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

# Đọc video
video_path = "mikazuki.mp4"
cap = cv2.VideoCapture(video_path)
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
counted_ids = set()
vehicle_count = 0

# Thiết lập video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "output_video.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

cv2.namedWindow("Vehicle Tracking and Counting")
cv2.setMouseCallback("Vehicle Tracking and Counting", draw_line_and_arrow)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Giai đoạn thiết lập
        if not setup_done:
            frame_copy = frame.copy()
            if line_start is not None:
                cv2.line(frame_copy, line_start, line_start if line_end is None else line_end, (0, 0, 255), 2)
            if arrow_start is not None:
                cv2.arrowedLine(frame_copy, arrow_start, arrow_start if arrow_end is None else arrow_end, (255, 0, 0), 2)
            cv2.putText(frame_copy, "Draw line (red) then arrow (blue) for direction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Vehicle Tracking and Counting", frame_copy)
            out.write(frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Lưu tọa độ line và arrow sau khi thiết lập xong
        if setup_done and frame_counter == 0:
            save_line_and_arrow_to_txt()

        # Tính vector hướng từ mũi tên
        direction_vector = (arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1])

        # Phát hiện và theo dõi
        results = model(frame, classes=vehicle_classes, conf=0.5)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, score, cls_id in zip(boxes, scores, class_ids):
                detections.append([*box, score, cls_id])
        detections = np.array(detections)

        if len(detections) > 0:
            tracks = tracker.update(detections, frame)
            for track in tracks:
                x1, y1, x2, y2, track_id, score, class_id, det_ind = track[:8]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Vẽ bounding box và thông tin
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"ID: {int(track_id)} | {model.names[int(class_id)]}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Kiểm tra xe cắt qua đường đếm và lưu frame
                if track_id in previous_positions:
                    prev_cx, prev_cy = previous_positions[track_id]
                    intersect = line_intersection(line_start, line_end, (prev_cx, prev_cy), (cx, cy))
                    if intersect and track_id not in counted_ids:
                        move_vec = (cx - prev_cx, cy - prev_cy)
                        dot_product = move_vec[0] * direction_vector[0] + move_vec[1] * direction_vector[1]
                        if dot_product > 0:
                            vehicle_count += 1
                            counted_ids.add(track_id)
                            # Lưu frame khi xe cắt qua line
                            frame_copy = frame.copy()
                            # Vẽ lại line và arrow lên frame trước khi lưu
                            cv2.line(frame_copy, line_start, line_end, (0, 0, 255), 2)
                            cv2.arrowedLine(frame_copy, arrow_start, arrow_end, (255, 0, 0), 2)
                            frame_path = os.path.join(output_frame_dir, f"vehicle_{frame_counter:04d}.jpg")
                            cv2.imwrite(frame_path, frame_copy)
                            frame_counter += 1
                            print(f"Đã lưu frame tại: {frame_path}")

                previous_positions[track_id] = (cx, cy)

        # Vẽ đường đếm và mũi tên
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
        cv2.arrowedLine(frame, arrow_start, arrow_end, (255, 0, 0), 2)
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        # Ghi frame vào video output
        out.write(frame)
        cv2.imshow("Vehicle Tracking and Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Giải phóng tài nguyên
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Total vehicles counted: {vehicle_count}")
    print(f"Video đã được lưu tại: {output_path}")
    print(f"Tổng số frame đã lưu: {frame_counter}")

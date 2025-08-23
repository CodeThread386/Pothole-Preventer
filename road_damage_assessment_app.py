import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.interpolate import make_interp_spline

# Load YOLOv8 model
best_model = YOLO("model/best.pt")

# Video input/output
video_path = "sample_video.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "road_damage_assessment.avi",
    fourcc,
    20.0,
    (int(cap.get(3)), int(cap.get(4))),
)

# Font
font = cv2.FONT_HERSHEY_SIMPLEX
num_parts = 5   # columns
num_rows = 4    # rows

# Smoothing buffers
damage_deques = [[deque(maxlen=20) for _ in range(num_parts)] for _ in range(num_rows)]

# Keep track of the yellow block's column (initialize to center)
if 'yellow_col' not in locals():
    yellow_col = num_parts // 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    part_width = w // num_parts
    row_height = (h // 2) // num_rows
    lower_half_start = h // 2

    smoothed_damages = np.zeros((num_rows, num_parts), dtype=np.float32)

    # --- Run YOLO & calculate smoothed damages ---
    for row in range(num_rows):
        for col in range(num_parts):
            x_start = col * part_width
            x_end = (col + 1) * part_width if col < num_parts - 1 else w
            y_start = lower_half_start + row * row_height
            y_end = lower_half_start + (row + 1) * row_height if row < num_rows - 1 else h
            square = frame[y_start:y_end, x_start:x_end]

            results = best_model.predict(source=square, imgsz=640, conf=0.25, verbose=False)

            percentage_damage = 0
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                image_area = square.shape[0] * square.shape[1]

                total_area = 0
                for mask in masks:
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        total_area += cv2.contourArea(c)

                if image_area > 0:
                    percentage_damage = (total_area / image_area) * 100

            damage_deques[row][col].append(percentage_damage)
            smoothed_damages[row, col] = np.mean(damage_deques[row][col])

    # --- Dynamic programming for smooth path ---
    dp = np.full((num_rows, num_parts), np.inf)
    prev = np.full((num_rows, num_parts), -1, dtype=int)

    dp[0, :] = smoothed_damages[0, :]

    for row in range(1, num_rows):
        for col in range(num_parts):
            # only allow small turns (left, straight, right)
            for d in [-1, 0, 1]:
                prev_col = col + d
                if 0 <= prev_col < num_parts:
                    # add small penalty to encourage straight path
                    penalty = 0.5 if d != 0 else 0
                    val = dp[row - 1, prev_col] + smoothed_damages[row, col] + penalty
                    if val < dp[row, col]:
                        dp[row, col] = val
                        prev[row, col] = prev_col

    # Reconstruct path
    min_sum_col = np.argmin(dp[num_rows - 1, :])
    path = [0] * num_rows
    path[-1] = min_sum_col
    for row in range(num_rows - 1, 0, -1):
        path[row - 1] = prev[row, path[row]]

    # --- Yellow block movement restriction ---
    # Only allow yellow_col to move to adjacent or same column as previous yellow_col
    target_col = path[-2]
    possible_moves = [yellow_col]
    if yellow_col > 0:
        possible_moves.append(yellow_col - 1)
    if yellow_col < num_parts - 1:
        possible_moves.append(yellow_col + 1)
    # Move to the adjacent position (from possible_moves) that is closest to the target_col
    yellow_col = min(possible_moves, key=lambda c: abs(c - target_col))

    # --- Draw overlays ---
    processed_frame = frame.copy()
    overlay = processed_frame.copy()

    for row in range(num_rows):
        for col in range(num_parts):
            x_start = col * part_width
            x_end = (col + 1) * part_width if col < num_parts - 1 else w
            y_start = lower_half_start + row * row_height
            y_end = lower_half_start + (row + 1) * row_height if row < num_rows - 1 else h

            # Highlight the needed tile in the last layer with yellow (only one tile moves at a time)
            if row == num_rows - 1 and col == yellow_col:
                overlay_color, alpha = (0, 255, 255), 0.25  # yellow
            elif col == path[row]:
                overlay_color, alpha = (0, 200, 0), 0.15  # selected path (green)
            else:
                overlay_color, alpha = (0, 0, 255), 0.08  # default red

            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), overlay_color, -1)

            cv2.putText(processed_frame,
                        f"{smoothed_damages[row, col]:.2f}%",
                        (x_start + 10, y_start + 25),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw smooth curve
    pts = np.array(list(zip([(col * part_width + part_width // 2) for col in path],
                            [lower_half_start + row * row_height + row_height // 2 for row in range(num_rows)])), np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(processed_frame, [pts], False, (0, 255, 0), 3)

    cv2.addWeighted(overlay, 0.4, processed_frame, 0.6, 0, processed_frame)

    # Save & display
    out.write(processed_frame)
    cv2.imshow("Road Damage Assessment", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

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

def render_heatmap(smoothed_grid: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Make a visible heatmap from smoothed damages.
    Uses percentile-based normalization for stable contrast.
    """
    grid = smoothed_grid.astype(np.float32)

    # Robust min/max to avoid flat maps
    vmin = float(np.percentile(grid, 5))
    vmax = float(np.percentile(grid, 95))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
    norm8 = (norm * 255).astype(np.uint8)            # shape (rows, cols)
    # Resize to lower-half resolution (width x height)
    hm = cv2.resize(norm8, (width, height), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.GaussianBlur(hm, (0, 0), 1.0)           # mild smoothing
    return hm

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

            percentage_damage = 0.0
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                image_area = square.shape[0] * square.shape[1]
                total_area = 0.0
                for mask in masks:
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        total_area += cv2.contourArea(c)
                if image_area > 0:
                    percentage_damage = (total_area / image_area) * 100.0

            damage_deques[row][col].append(percentage_damage)
            smoothed_damages[row, col] = float(np.mean(damage_deques[row][col]))

    # --- Dynamic programming with curvature penalty ---
    dp = np.full((num_rows, num_parts), np.inf, dtype=np.float32)
    prev = np.full((num_rows, num_parts), -1, dtype=int)
    dp[0, :] = smoothed_damages[0, :]

    for row in range(1, num_rows):
        for col in range(num_parts):
            for d in [-1, 0, 1]:
                prev_col = col + d
                if 0 <= prev_col < num_parts:
                    base_cost = smoothed_damages[row, col]
                    turn_penalty = 0.5 if d != 0 else 0.0
                    if row > 1 and prev[row - 1, prev_col] != -1:
                        curvature = abs(prev_col - prev[row - 1, prev_col])
                        curvature_penalty = curvature * 0.7
                    else:
                        curvature_penalty = 0.0

                    val = dp[row - 1, prev_col] + base_cost + turn_penalty + curvature_penalty
                    if val < dp[row, col]:
                        dp[row, col] = val
                        prev[row, col] = prev_col

    # --- Reconstruct path ---
    min_sum_col = int(np.argmin(dp[num_rows - 1, :]))
    path = [0] * num_rows
    path[-1] = min_sum_col
    for row in range(num_rows - 1, 0, -1):
        path[row - 1] = int(prev[row, path[row]])

    # --- Interpolate for smoother path ---
    x_coords = [(col * part_width + part_width // 2) for col in path]
    y_coords = [lower_half_start + row * row_height + row_height // 2 for row in range(num_rows)]

    if len(x_coords) >= 3:
        spline = make_interp_spline(y_coords, x_coords, k=2)
        y_smooth = np.linspace(min(y_coords), max(y_coords), 100)
        x_smooth = spline(y_smooth)
        smooth_pts = np.array(list(zip(x_smooth, y_smooth)), np.int32).reshape((-1, 1, 2))
    else:
        smooth_pts = np.array(list(zip(x_coords, y_coords)), np.int32).reshape((-1, 1, 2))

    # --- Yellow block logic ---
    target_col = path[-2]
    possible_moves = [yellow_col]
    if yellow_col > 0: possible_moves.append(yellow_col - 1)
    if yellow_col < num_parts - 1: possible_moves.append(yellow_col + 1)
    yellow_col = min(possible_moves, key=lambda c: abs(c - target_col))

    # --- Compose frame: heatmap first, then overlays ---
    processed_frame = frame.copy()

    # Heatmap only on lower half
    heatmap_color = render_heatmap(smoothed_damages, w, h - lower_half_start)
    heat_alpha = 0.45
    processed_frame[lower_half_start:h, 0:w] = cv2.addWeighted(
        processed_frame[lower_half_start:h, 0:w], 1 - heat_alpha,
        heatmap_color, heat_alpha, 0
    )

    # Now draw grid overlays on top (keep them subtle so heatmap shows through)
    overlay = processed_frame.copy()

    for row in range(num_rows):
        for col in range(num_parts):
            x_start = col * part_width
            x_end = (col + 1) * part_width if col < num_parts - 1 else w
            y_start = lower_half_start + row * row_height
            y_end = lower_half_start + (row + 1) * row_height if row < num_rows - 1 else h

            # Make the last yellow tile bright and high exposure yellow
            if row == num_rows - 1 and col == yellow_col:
                overlay_color, alpha = (0, 255, 255), 0.85  # bright yellow, high alpha
            elif col == path[row]:
                overlay_color, alpha = (0, 200, 0), 0.15    # green path
            else:
                overlay_color, alpha = (0, 0, 255), 0.06    # subtle red

            rect_layer = processed_frame.copy()
            cv2.rectangle(rect_layer, (x_start, y_start), (x_end, y_end), overlay_color, -1)
            processed_frame = cv2.addWeighted(rect_layer, alpha, processed_frame, 1 - alpha, 0)

            # Text on top (skip for transparent tile if you want)
            cv2.putText(processed_frame,
                        f"{smoothed_damages[row, col]:.2f}%",
                        (x_start + 10, y_start + 25),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw smooth curve last
    cv2.polylines(processed_frame, [smooth_pts], False, (0, 255, 0), 3)

    # Save & display
    out.write(processed_frame)
    cv2.imshow("Road Damage Assessment with Heatmap", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

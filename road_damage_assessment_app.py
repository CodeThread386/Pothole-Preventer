import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.interpolate import make_interp_spline
import torch
import time

# ---------------- CONFIG (tweak these) ----------------
MODEL_PATH = "model/best.pt"     # your segmentation model
VIDEO_SOURCE = "sample_video.mp4"                 # 0 for webcam, or "sample_video.mp4"
IMG_SZ = 640
CONF = 0.25
FPS = 20.0

num_parts = 5      # columns
num_rows = 4       # rows
DEQUE_LEN = 20     # smoothing of percentage per tile

# Online planning / smoothness
HORIZON = 10               # MPC horizon in frames (how far ahead to plan)
DISCOUNT = 0.98            # discount factor for future cost (makes planning short-sighted)
TURN_PENALTY = 0.5         # penalty for lateral move (same as original)
CURV_PENALTY_FACTOR = 0.7  # curvature penalty factor (same as original)
EMA_HEAT_ALPHA = 0.25      # heatmap EMA smoothing (lower -> smoother)
EMA_PATH_ALPHA = 0.25      # path x-coord EMA smoothing (lower -> smoother)
MAX_COL_CHANGE_PER_FRAME = 1  # allowed column change per frame (vehicle constraint)
WARMUP = True
# -----------------------------------------------------

# Load model and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
try:
    model.to(device)
except Exception:
    pass  # some ultralytics builds accept device in predict

# open video (file or webcam)
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("Cannot open video source")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_w == 0 or frame_h == 0:
    # fallback sizes for some webcams; try reading one frame
    ret, tmp = cap.read()
    if not ret:
        raise SystemExit("Cannot read frame to determine size")
    frame_h, frame_w = tmp.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Precompute tile geometry (lower half)
part_width = frame_w // num_parts
row_height = (frame_h // 2) // num_rows
lower_half_start = frame_h // 2

tile_bounds = []
tile_areas = []
for row in range(num_rows):
    for col in range(num_parts):
        x_start = col * part_width
        x_end = (col + 1) * part_width if col < num_parts - 1 else frame_w
        y_start = lower_half_start + row * row_height
        y_end = lower_half_start + (row + 1) * row_height if row < num_rows - 1 else frame_h
        tile_bounds.append((row, col, x_start, x_end, y_start, y_end))
        tile_areas.append(max(1, (y_end - y_start) * (x_end - x_start)))

# smoothing structures
damage_deques = [[deque(maxlen=DEQUE_LEN) for _ in range(num_parts)] for _ in range(num_rows)]
ema_heatmap = None            # EMA-smoothed grid used for DP
ema_path_x = None             # EMA-smoothed x positions for trajectory drawing

# yellow tile tracked column (initialized center)
yellow_col = num_parts // 2

# warm up model once to reduce first-call latency
if WARMUP:
    dummy = np.zeros((IMG_SZ // 2, IMG_SZ // 2, 3), dtype=np.uint8)
    try:
        _ = model.predict(source=dummy, imgsz=IMG_SZ, conf=CONF, device=device, verbose=False)
    except Exception:
        try:
            _ = model.predict(source=dummy, imgsz=IMG_SZ, conf=CONF, verbose=False)
        except Exception:
            pass

def render_heatmap(smoothed_grid: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = smoothed_grid.astype(np.float32)
    vmin = float(np.percentile(grid, 5))
    vmax = float(np.percentile(grid, 95))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
    norm8 = (norm * 255).astype(np.uint8)
    hm = cv2.resize(norm8, (width, height), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.GaussianBlur(hm, (0, 0), 1.0)
    return hm

def mpc_next_col(current_col, last_row_cost, horizon=HORIZON):
    """
    Receding-horizon DP (MPC): for given current_col and cost vector for last-row (shape: num_parts),
    compute optimal column sequence for horizon steps minimizing discounted cumulative cost,
    with adjacency constraint (±1). Return next column (first step).
    """
    # Build horizon costs: use same last_row_cost each future step with discount;
    # this is a simple prediction strategy; you can replace with better forecast.
    costs = np.zeros((horizon, num_parts), dtype=np.float32)
    for t in range(horizon):
        costs[t, :] = (DISCOUNT ** t) * last_row_cost  # discounted future cost

    # dp[t,c] minimal cost up to time t (0-indexed) ending at column c
    dp = np.full((horizon, num_parts), np.inf, dtype=np.float32)
    prev = np.full((horizon, num_parts), -1, dtype=int)

    # t = 0: can move from current_col to current_col or ±1?
    for c in range(num_parts):
        if abs(c - current_col) <= MAX_COL_CHANGE_PER_FRAME:
            # allow immediate first transition within vehicle constraint
            dp[0, c] = costs[0, c]
            prev[0, c] = current_col

    # subsequent times
    for t in range(1, horizon):
        for c in range(num_parts):
            # allowed previous columns are c-1,c,c+1 (adjacent)
            best_val = np.inf
            best_p = -1
            for p in (c - 1, c, c + 1):
                if 0 <= p < num_parts:
                    # penalize lateral change relative to prev step
                    move = abs(c - p)
                    turn_pen = TURN_PENALTY if move != 0 else 0.0
                    val = dp[t - 1, p] + costs[t, c] + turn_pen
                    # curvature penalty: if t>1 we can approximate by second difference if prev prev exists
                    if t > 1 and prev[t - 1, p] != -1:
                        curvature = abs(p - prev[t - 1, p])
                        val += curvature * CURV_PENALTY_FACTOR
                    if val < best_val:
                        best_val = val
                        best_p = p
            dp[t, c] = best_val
            prev[t, c] = best_p

    # reconstruct best end column and backtrack to get sequence
    end_col = int(np.argmin(dp[horizon - 1, :]))
    seq = [0] * horizon
    seq[horizon - 1] = end_col
    for t in range(horizon - 1, 0, -1):
        seq[t - 1] = int(prev[t, seq[t]])

    # the first step (t=0) is seq[0] (reachable within MAX_COL_CHANGE_PER_FRAME)
    return seq[0], seq

# Main live loop
print("Starting live processing. Press 'q' to quit.")
prev_time = time.time()
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # ROI = lower half
    roi = frame[lower_half_start:frame_h, :]

    # Run YOLO once on ROI for segmentation masks
    try:
        results = model.predict(source=roi, imgsz=IMG_SZ, conf=CONF, device=device, verbose=False)
    except TypeError:
        results = model.predict(source=roi, imgsz=IMG_SZ, conf=CONF, verbose=False)
    except Exception:
        results = []

    # accumulate mask pixels per tile
    tile_mask_pixels = np.zeros((num_rows, num_parts), dtype=np.float32)
    if results and len(results) > 0 and getattr(results[0], "masks", None) is not None:
        try:
            masks_np = results[0].masks.data.cpu().numpy()
        except Exception:
            masks_np = np.array(results[0].masks.data)

        roi_h, roi_w = roi.shape[:2]
        for mask in masks_np:
            # threshold and resize to roi if necessary
            if mask.shape[0] != roi_h or mask.shape[1] != roi_w:
                mask_r = cv2.resize((mask > 0).astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_r = (mask > 0).astype(np.uint8)

            for r in range(num_rows):
                y0 = r * row_height
                y1 = (r + 1) * row_height if r < num_rows - 1 else roi_h
                y1 = min(y1, roi_h)
                if y0 >= y1:
                    continue
                for c in range(num_parts):
                    x0 = c * part_width
                    x1 = (c + 1) * part_width if c < num_parts - 1 else roi_w
                    x1 = min(x1, roi_w)
                    if x0 >= x1:
                        continue
                    cell_mask = mask_r[y0:y1, x0:x1]
                    if cell_mask.size == 0:
                        continue
                    tile_mask_pixels[r, c] += int(cell_mask.sum())

    # convert to percentage and append to deques (preserve original deque smoothing)
    current_grid = np.zeros((num_rows, num_parts), dtype=np.float32)
    for idx, (row, col, x0, x1, y0, y1) in enumerate(tile_bounds):
        area = tile_areas[idx]
        count = tile_mask_pixels[row, col]
        pct = (count / float(area)) * 100.0 if area > 0 else 0.0
        damage_deques[row][col].append(pct)
        current_grid[row, col] = float(np.mean(damage_deques[row][col]))

    # EMA heatmap smoothing for temporal stability
    if ema_heatmap is None:
        ema_heatmap = current_grid.copy()
    else:
        ema_heatmap = EMA_HEAT_ALPHA * current_grid + (1.0 - EMA_HEAT_ALPHA) * ema_heatmap

    smoothed_damages = ema_heatmap  # use this for DP & visualization

    # Build per-row DP path across rows (like original) to compute path positions across rows
    # We'll compute the 'path' column per row minimizing smoothed damages (identical to original)
    dp_rows = np.full((num_rows, num_parts), np.inf, dtype=np.float32)
    prev_rows = np.full((num_rows, num_parts), -1, dtype=int)
    dp_rows[0, :] = smoothed_damages[0, :]

    for r in range(1, num_rows):
        for c in range(num_parts):
            for d in (-1, 0, 1):
                p = c + d
                if 0 <= p < num_parts:
                    base_cost = smoothed_damages[r, c]
                    turn_penalty = 0.5 if d != 0 else 0.0
                    if r > 1 and prev_rows[r - 1, p] != -1:
                        curvature = abs(p - prev_rows[r - 1, p])
                        curvature_penalty = curvature * 0.7
                    else:
                        curvature_penalty = 0.0
                    val = dp_rows[r - 1, p] + base_cost + turn_penalty + curvature_penalty
                    if val < dp_rows[r, c]:
                        dp_rows[r, c] = val
                        prev_rows[r, c] = p

    # reconstruct per-row path (col indices)
    end_col = int(np.argmin(dp_rows[num_rows - 1, :]))
    path_cols = [0] * num_rows
    path_cols[-1] = end_col
    for r in range(num_rows - 1, 0, -1):
        path_cols[r - 1] = int(prev_rows[r, path_cols[r]])

    # compute center x for each row from path_cols (float)
    x_centers = np.array([(col * part_width + part_width / 2.0) for col in path_cols], dtype=np.float32)
    y_centers = np.array([lower_half_start + r * row_height + row_height / 2.0 for r in range(num_rows)], dtype=np.float32)

    # EMA on path x positions for lateral smoothing over time
    if ema_path_x is None:
        ema_path_x = x_centers.copy()
    else:
        ema_path_x = EMA_PATH_ALPHA * x_centers + (1.0 - EMA_PATH_ALPHA) * ema_path_x

    # Use MPC on last-row smoothed damages to get next yellow column suggestion
    last_row_cost = smoothed_damages[num_rows - 1, :]
    recommended_col, planned_seq = mpc_next_col(yellow_col, last_row_cost, horizon=HORIZON)

    # enforce vehicle discrete move constraint: change at most MAX_COL_CHANGE_PER_FRAME
    desired_move = np.clip(recommended_col - yellow_col, -MAX_COL_CHANGE_PER_FRAME, MAX_COL_CHANGE_PER_FRAME)
    yellow_col = int(yellow_col + desired_move)

    # Compose frame for visualization
    processed = frame.copy()
    heatmap_color = render_heatmap(smoothed_damages, frame_w, frame_h - lower_half_start)
    heat_alpha = 0.45
    processed[lower_half_start:frame_h, 0:frame_w] = cv2.addWeighted(
        processed[lower_half_start:frame_h, 0:frame_w], 1 - heat_alpha,
        heatmap_color, heat_alpha, 0
    )

    # Draw grid overlays and text
    for r in range(num_rows):
        for c in range(num_parts):
            x_start = c * part_width
            x_end = (c + 1) * part_width if c < num_parts - 1 else frame_w
            y_start = lower_half_start + r * row_height
            y_end = lower_half_start + (r + 1) * row_height if r < num_rows - 1 else frame_h

            if r == num_rows - 1 and c == yellow_col:
                overlay_color, alpha = (0, 255, 255), 0.85
            elif c == path_cols[r]:
                overlay_color, alpha = (0, 200, 0), 0.15
            else:
                overlay_color, alpha = (0, 0, 255), 0.06

            rect_layer = processed.copy()
            cv2.rectangle(rect_layer, (x_start, y_start), (x_end, y_end), overlay_color, -1)
            processed = cv2.addWeighted(rect_layer, alpha, processed, 1 - alpha, 0)

            cv2.putText(processed, f"{smoothed_damages[r, c]:.2f}%", (x_start + 10, y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw smooth spline for current ema_path_x
    try:
        if len(ema_path_x) >= 3:
            k = min(3, len(ema_path_x) - 1)
            spline = make_interp_spline(y_centers, ema_path_x, k=k)
            ys = np.linspace(y_centers.min(), y_centers.max(), 200)
            xs = spline(ys)
            pts = np.array(list(zip(xs, ys)), np.int32).reshape((-1, 1, 2))
            cv2.polylines(processed, [pts], False, (0, 255, 0), 4, lineType=cv2.LINE_AA)
        else:
            pts = np.array(list(zip(ema_path_x, y_centers)), np.int32).reshape((-1, 1, 2))
            cv2.polylines(processed, [pts], False, (0, 255, 0), 4, lineType=cv2.LINE_AA)
    except Exception:
        pass

    # Draw small indicator and planned immediate sequence (for debugging)
    cv2.putText(processed, f"Frame: {frame_idx}  Yellow col: {yellow_col}  Recommended: {recommended_col}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Optionally show planned sequence for next horizon on the bottom row
    seq_text = "Planned: " + ",".join(str(x) for x in planned_seq[:min(len(planned_seq), 10)])
    cv2.putText(processed, seq_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Live Heatmap + MPC Yellow Path", processed)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # small controlled sleep to match FPS (if reading from file, remove)
    now = time.time()
    elapsed = now - prev_time
    prev_time = now
    # don't block if camera; just continue

cap.release()
cv2.destroyAllWindows() 
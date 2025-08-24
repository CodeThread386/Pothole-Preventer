import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.interpolate import make_interp_spline
import torch
import time

# ---------------- CONFIG (tweak these) ----------------
MODEL_PATH = "model/best.pt"     # segmentation model
VIDEO_SOURCE = "sample_video.mp4"  # 0 for webcam
IMG_SZ = 640
CONF = 0.25

num_parts = 5
num_rows = 4
DEQUE_LEN = 20

# MPC / smoothing
HORIZON = 10
DISCOUNT = 0.98
TURN_PENALTY = 0.5
CURV_PENALTY_FACTOR = 0.7
EMA_HEAT_ALPHA = 0.25
EMA_PATH_ALPHA = 0.25
MAX_COL_CHANGE_PER_FRAME = 1    # keep car-feasible (discrete columns per frame)
WARMUP = True

# Speed estimation and dynamic ROI
REAL_DISTANCE_M = 10.0
LINE_FAR_FRAC = 0.30
LINE_NEAR_FRAC = 0.70
SPEED_TO_ROI_MAP = [
    (20.0, 0.25),
    (60.0, 0.40),
    (9999.0, 0.60)
]

# Danger avoidance weight (higher -> avoid danger more aggressively)
DANGER_WEIGHT = 1.5   # tune between 0.0 (ignore) and ~2.0
# row weighting for danger scoring (nearer rows matter more)
ROW_WEIGHTS = np.linspace(1.5, 0.6, num_rows)  # length num_rows
# -----------------------------------------------------

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
try:
    model.to(device)
except Exception:
    pass

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("Cannot open video source")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_est = cap.get(cv2.CAP_PROP_FPS) or 20.0
if frame_w == 0 or frame_h == 0:
    ret, tmp = cap.read()
    if not ret:
        raise SystemExit("Cannot read frame to determine size")
    frame_h, frame_w = tmp.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

part_width = frame_w // num_parts

# smoothing structures
damage_deques = [[deque(maxlen=DEQUE_LEN) for _ in range(num_parts)] for _ in range(num_rows)]
ema_heatmap = None
ema_path_x = None

# yellow tile (vehicle) state
yellow_col = num_parts // 2                  # discrete column vehicle currently in (logical)
yellow_x_pos = (yellow_col * part_width + part_width / 2.0)  # pixel x (float) for trail
yellow_history = []                          # list of (x,y) points for trail

# display_path_cols for natural planned visualization (keeps tiles changing smoothly)
display_path_cols = [yellow_col] * num_rows

# warm up
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
    costs = np.zeros((horizon, num_parts), dtype=np.float32)
    for t in range(horizon):
        costs[t, :] = (DISCOUNT ** t) * last_row_cost
    dp = np.full((horizon, num_parts), np.inf, dtype=np.float32)
    prev = np.full((horizon, num_parts), -1, dtype=int)
    for c in range(num_parts):
        if abs(c - current_col) <= MAX_COL_CHANGE_PER_FRAME:
            dp[0, c] = costs[0, c]
            prev[0, c] = current_col
    for t in range(1, horizon):
        for c in range(num_parts):
            best_val = np.inf
            best_p = -1
            for p in (c - 1, c, c + 1):
                if 0 <= p < num_parts:
                    move = abs(c - p)
                    turn_pen = TURN_PENALTY if move != 0 else 0.0
                    val = dp[t - 1, p] + costs[t, c] + turn_pen
                    if t > 1 and prev[t - 1, p] != -1:
                        curvature = abs(p - prev[t - 1, p])
                        val += curvature * CURV_PENALTY_FACTOR
                    if val < best_val:
                        best_val = val
                        best_p = p
            dp[t, c] = best_val
            prev[t, c] = best_p
    end_col = int(np.argmin(dp[horizon - 1, :]))
    seq = [0] * horizon
    seq[horizon - 1] = end_col
    for t in range(horizon - 1, 0, -1):
        seq[t - 1] = int(prev[t, seq[t]])
    return seq[0], seq

# optical flow storage
prev_roi_gray = None

def estimate_speed_from_flow(prev_gray, cur_gray, roi_h):
    if prev_gray is None or cur_gray is None or prev_gray.shape != cur_gray.shape:
        return 0.0
    y_far = int(roi_h * LINE_FAR_FRAC)
    y_near = int(roi_h * LINE_NEAR_FRAC)
    pixel_dist = max(1.0, abs(y_near - y_far))
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
    band = flow[y_far:y_near, :, 1]
    if band.size == 0:
        return 0.0
    mean_v_pixels_per_frame = np.nanmean(band)
    v_pixels_per_s = abs(mean_v_pixels_per_frame) * (fps_est or 20.0)
    if v_pixels_per_s < 1e-3:
        return 0.0
    time_to_cross = pixel_dist / v_pixels_per_s
    if time_to_cross <= 0:
        return 0.0
    speed_m_s = REAL_DISTANCE_M / time_to_cross
    return float(speed_m_s * 3.6)

def choose_roi_fraction_for_speed(speed_kmph):
    for thresh, frac in SPEED_TO_ROI_MAP:
        if speed_kmph < thresh:
            return frac
    return SPEED_TO_ROI_MAP[-1][1]

print("Starting live processing.")
prev_time = time.time()
frame_idx = 0
speed_kmph = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Bootstrap ROI for flow estimate
    if prev_roi_gray is None:
        roi_frac = 0.40
    else:
        roi_frac = 0.40

    roi_h_tmp = int(frame_h * roi_frac)
    roi_top_tmp = max(0, frame_h - roi_h_tmp)
    roi_tmp = frame[roi_top_tmp:frame_h, :]
    cur_tmp_gray = cv2.cvtColor(roi_tmp, cv2.COLOR_BGR2GRAY)

    if prev_roi_gray is not None and prev_roi_gray.shape == cur_tmp_gray.shape:
        speed_kmph = estimate_speed_from_flow(prev_roi_gray, cur_tmp_gray, cur_tmp_gray.shape[0])
    else:
        speed_kmph = 0.0

    roi_frac = choose_roi_fraction_for_speed(speed_kmph)
    roi_height = int(frame_h * roi_frac)
    roi_top = max(0, frame_h - roi_height)
    roi = frame[roi_top:frame_h, :]

    cur_roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if prev_roi_gray is not None and prev_roi_gray.shape == cur_roi_gray.shape:
        speed_kmph = estimate_speed_from_flow(prev_roi_gray, cur_roi_gray, cur_roi_gray.shape[0])
    prev_roi_gray = cur_roi_gray.copy()

    # segmentation on ROI
    try:
        results = model.predict(source=roi, imgsz=IMG_SZ, conf=CONF, device=device, verbose=False)
    except TypeError:
        results = model.predict(source=roi, imgsz=IMG_SZ, conf=CONF, verbose=False)
    except Exception:
        results = []

    # recompute row_height because ROI changed
    row_height = max(1, roi.shape[0] // num_rows)

    # accumulate mask pixels per cell
    tile_mask_pixels = np.zeros((num_rows, num_parts), dtype=np.float32)
    if results and len(results) > 0 and getattr(results[0], "masks", None) is not None:
        try:
            masks_np = results[0].masks.data.cpu().numpy()
        except Exception:
            masks_np = np.array(results[0].masks.data)

        roi_h, roi_w = roi.shape[:2]
        for mask in masks_np:
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

    # convert to percentages and update deques
    current_grid = np.zeros((num_rows, num_parts), dtype=np.float32)
    for r in range(num_rows):
        for c in range(num_parts):
            y0 = r * row_height
            y1 = (r + 1) * row_height if r < num_rows - 1 else roi.shape[0]
            tile_w = ((c + 1) * part_width if c < num_parts - 1 else roi.shape[1]) - (c * part_width)
            tile_h = y1 - y0
            area = max(1, tile_w * tile_h)
            cnt = tile_mask_pixels[r, c]
            pct = (cnt / float(area)) * 100.0 if area > 0 else 0.0
            damage_deques[r][c].append(pct)
            current_grid[r, c] = float(np.mean(damage_deques[r][c]))

    # EMA heatmap smoothing
    if ema_heatmap is None:
        ema_heatmap = current_grid.copy()
    else:
        ema_heatmap = EMA_HEAT_ALPHA * current_grid + (1.0 - EMA_HEAT_ALPHA) * ema_heatmap
    smoothed_damages = ema_heatmap

    # DP across rows (same path logic)
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

    # reconstruct per-row path columns
    end_col = int(np.argmin(dp_rows[num_rows - 1, :]))
    path_cols = [0] * num_rows
    path_cols[-1] = end_col
    for r in range(num_rows - 1, 0, -1):
        path_cols[r - 1] = int(prev_rows[r, path_cols[r]])

    # compute danger scores per column (row-weighted sum)
    # higher value -> more pothole danger in that column ahead
    danger_scores = np.sum(smoothed_damages * ROW_WEIGHTS[:, None], axis=0)  # shape (num_parts,)

    # Last-row cost (as previous) -> we'll bias it with danger scores
    last_row_cost = smoothed_damages[num_rows - 1, :]

    # Danger-aware modified last-row cost for MPC (optionally used by planner)
    danger_modified_cost = last_row_cost + DANGER_WEIGHT * danger_scores

    # Use MPC (unchanged inputs if you want) but here we use danger_modified_cost so planner avoids danger columns
    recommended_col, planned_seq = mpc_next_col(yellow_col, danger_modified_cost, horizon=HORIZON)

    # Now decide the actual discrete next column for the vehicle.
    # Candidate move MUST be within +/- MAX_COL_CHANGE_PER_FRAME for safety.
    # We'll consider feasible candidates and choose the one with lowest combined danger+cost.
    feasible = []
    for cand in range(max(0, yellow_col - MAX_COL_CHANGE_PER_FRAME), min(num_parts, yellow_col + MAX_COL_CHANGE_PER_FRAME + 1)):
        # score combines danger (ahead) and the dp cost for that column (last_row cost) and slight turn penalty
        score = danger_scores[cand] * DANGER_WEIGHT + last_row_cost[cand] + 0.2 * abs(cand - yellow_col)
        feasible.append((score, cand))
    # choose best feasible
    feasible.sort()
    chosen_score, chosen_col = feasible[0]

    # apply sudden (non-smooth) discrete move to chosen_col (but still restricted to +/- MAX_COL_CHANGE_PER_FRAME)
    yellow_col = chosen_col

    # set yellow_x_pos to be exact center of yellow_col (no smooth transition)
    yellow_x_pos = yellow_col * part_width + part_width / 2.0

    # append to history (visual trail)
    yellow_history.append((int(yellow_x_pos), frame_h - 4))
    if len(yellow_history) > 200:
        yellow_history.pop(0)

    # update display_path_cols gradually to keep planned visuals natural (±1 per frame)
    for r in range(num_rows):
        desired = int(path_cols[r])
        cur = int(display_path_cols[r])
        if desired > cur:
            display_path_cols[r] = cur + 1
        elif desired < cur:
            display_path_cols[r] = cur - 1
        else:
            display_path_cols[r] = cur

    # Compose visualization (heatmap in lower half preserved)
    processed = frame.copy()
    heatmap_color = render_heatmap(smoothed_damages, frame_w, frame_h - (frame_h // 2))
    heat_alpha = 0.45
    lower_half_start = frame_h // 2
    processed[lower_half_start:frame_h, 0:frame_w] = cv2.addWeighted(
        processed[lower_half_start:frame_h, 0:frame_w], 1 - heat_alpha,
        heatmap_color, heat_alpha, 0
    )

    # Draw grid overlays (visual grid anchored to lower half) — uses per-pixel overlay so there are no seams
    vis_row_height = (frame_h // 2) // num_rows
    vis_lower_half_start = frame_h // 2

    # Build per-pixel overlay and alpha mask (so no seams / lines appear between tiles)
    overlay = np.zeros_like(processed, dtype=np.float32)     # BGR float
    alpha_mask = np.zeros((processed.shape[0], processed.shape[1], 1), dtype=np.float32)  # single-channel alpha

    for r in range(num_rows):
        for c in range(num_parts):
            x_start = int(c * part_width)
            x_end = int((c + 1) * part_width) if c < num_parts - 1 else frame_w
            y_start = int(vis_lower_half_start + r * vis_row_height)
            y_end = int(vis_lower_half_start + (r + 1) * vis_row_height) if r < num_rows - 1 else frame_h

            # Decide color & alpha per tile (no drawing of borders)
            if r == num_rows - 1 and c == yellow_col:
                # Yellow tile: color only from the tile's vertical CENTER down to bottom
                color = (0, 255, 255)  # BGR yellow
                tile_alpha = 0.85
                y_mid = (y_start + y_end) // 2
                overlay[y_mid:y_end, x_start:x_end, :] = color
                alpha_mask[y_mid:y_end, x_start:x_end, 0] = tile_alpha
            else:
                # default danger tile (red, very low alpha) for all other tiles
                color = (0, 0, 255)
                tile_alpha = 0.06
                overlay[y_start:y_end, x_start:x_end, :] = color
                alpha_mask[y_start:y_end, x_start:x_end, 0] = tile_alpha

    # Now blend overlay onto processed using per-pixel alpha (no seams)
    proc_f = processed.astype(np.float32)
    blended = (proc_f * (1.0 - alpha_mask)) + (overlay * alpha_mask)
    processed = np.clip(blended, 0, 255).astype(np.uint8)

    # Draw yellow trail (unchanged)
    for hx, hy in yellow_history[-60:]:
        cv2.circle(processed, (hx, hy), 3, (0, 200, 200), -1, lineType=cv2.LINE_AA)

    # Show the frame (no HUD text)
    cv2.imshow("Processed Screening", processed)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    now = time.time()
    elapsed = now - prev_time
    prev_time = now

cap.release()
cv2.destroyAllWindows()

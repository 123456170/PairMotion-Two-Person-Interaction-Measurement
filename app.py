import base64
import json
import logging
import math
import os
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pairmotion.app")


SEED = 20261012
FPS = 30
DURATION_S = 12.0

CAM_W, CAM_H = 960, 540
TD_W, TD_H = 540, 540

JOINT_NAMES = [
    "pelvis",
    "spine",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
    "root",
]

J = {name: idx for idx, name in enumerate(JOINT_NAMES)}

BONES = [
    (J["pelvis"], J["spine"]),
    (J["spine"], J["neck"]),
    (J["neck"], J["head"]),
    (J["spine"], J["l_shoulder"]),
    (J["l_shoulder"], J["l_elbow"]),
    (J["l_elbow"], J["l_wrist"]),
    (J["spine"], J["r_shoulder"]),
    (J["r_shoulder"], J["r_elbow"]),
    (J["r_elbow"], J["r_wrist"]),
    (J["pelvis"], J["l_hip"]),
    (J["l_hip"], J["l_knee"]),
    (J["l_knee"], J["l_ankle"]),
    (J["pelvis"], J["r_hip"]),
    (J["r_hip"], J["r_knee"]),
    (J["r_knee"], J["r_ankle"]),
]

DEFAULT_CONFIG = {
    "intimate_m": 0.45,
    "personal_m": 1.20,
    "social_m": 3.60,
}

# Four floor points used to create a metric homography.
# Meter coordinates are (x, z) on the floor plane, y=0.
FLOOR_METER_PTS = np.float32(
    [
        [-2.5, 1.0],
        [ 2.5, 1.0],
        [ 2.5, 7.0],
        [-2.5, 7.0],
    ]
)

# Matching image pixel coordinates for the synthetic camera view.
FLOOR_IMG_PTS = np.float32(
    [
        [260, 500],
        [700, 500],
        [610, 220],
        [350, 220],
    ]
)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = clamp01((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)


def angle_delta(a: float, b: float) -> float:
    """Signed shortest angular difference a-b in radians."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def euclidean(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def classify_zone(distance_m: float, cfg: Dict[str, float]) -> str:
    d = float(distance_m)
    if d < cfg["intimate_m"]:
        return "intimate"
    if d < cfg["personal_m"]:
        return "personal"
    if d < cfg["social_m"]:
        return "social"
    return "public"


def project_xz_batch(H: np.ndarray, xz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xz, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def project_xz(H: np.ndarray, x: float, z: float) -> np.ndarray:
    return project_xz_batch(H, np.array([[x, z]], dtype=np.float32))[0]


def local_scale(H: np.ndarray, x: float, z: float) -> float:
    """Pixels per meter near a floor point, estimated from a 0.5 m x-offset."""
    p0 = project_xz(H, x, z)
    p1 = project_xz(H, x + 0.5, z)
    return float(np.linalg.norm(p1 - p0) / 0.5)


def project_body_point(H: np.ndarray, x: float, z: float, y: float) -> np.ndarray:
    """
    Project a 3D body point using a floor homography plus a vertical pixel heuristic.
    This is synthetic/demo-grade, not a calibrated metric 3D reconstruction.
    """
    u, v = project_xz(H, x, z)
    s = local_scale(H, x, z)
    return np.array([u, v - y * s * 0.88], dtype=np.float32)


def project_joint_array(H: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """
    joints: (17, 3) array with columns x, y, z.
    Returns integer pixel coordinates (17, 2).
    """
    xz = joints[:, [0, 2]].astype(np.float32)
    floor_uv = project_xz_batch(H, xz).copy()
    uv = floor_uv.astype(np.float32)

    for k in range(joints.shape[0]):
        x = float(xz[k, 0])
        z = float(xz[k, 1])
        y = float(joints[k, 1])
        s = local_scale(H, x, z)
        uv[k, 1] = floor_uv[k, 1] - y * s * 0.88

    return np.round(uv).astype(np.int32)


def body_joints_world(
    x: float,
    z: float,
    theta: float,
    t: float,
    handshake: float,
    reach: float,
) -> np.ndarray:
    """
    Create a simple 17-joint body in world coordinates.
    theta is heading in the floor plane:
      forward = (sin(theta), cos(theta)) in (x, z)
      right   = (cos(theta), -sin(theta)) in (x, z)
    """
    coords = {}

    def set_joint(name: str, fwd: float, lat: float, y: float) -> None:
        fx = math.sin(theta)
        fz = math.cos(theta)
        rx = math.cos(theta)
        rz = -math.sin(theta)

        px = float(x) + fx * float(fwd) + rx * float(lat)
        pz = float(z) + fz * float(fwd) + rz * float(lat)
        coords[name] = (px, float(y), pz)

    gait = math.sin(2.0 * math.pi * 1.15 * t)
    gait *= 1.0 - 0.75 * float(handshake)

    set_joint("root", 0.0, 0.0, 0.0)
    set_joint("pelvis", 0.0, 0.0, 0.94)
    set_joint("spine", 0.02, 0.0, 1.15)
    set_joint("neck", 0.03, 0.0, 1.52)
    set_joint("head", 0.04, 0.0, 1.70)

    set_joint("l_shoulder", 0.02, -0.18, 1.40)
    set_joint("r_shoulder", 0.02, 0.18, 1.40)

    set_joint("l_elbow", -0.04 + 0.08 * gait, -0.24, 1.12)
    set_joint("r_elbow", -0.04 - 0.08 * gait, 0.24, 1.12)

    wrist_fwd = 0.08 + float(reach)
    wrist_lat = 0.22 - 0.12 * float(handshake)
    wrist_y = 0.95 + 0.12 * float(handshake)

    set_joint("l_wrist", wrist_fwd, -wrist_lat + 0.04 * gait, wrist_y)
    set_joint("r_wrist", wrist_fwd, wrist_lat - 0.04 * gait, wrist_y)

    set_joint("l_hip", 0.0, -0.10, 0.90)
    set_joint("r_hip", 0.0, 0.10, 0.90)

    set_joint("l_knee", 0.08 * gait, -0.11, 0.49)
    set_joint("r_knee", -0.08 * gait, 0.11, 0.49)

    set_joint("l_ankle", 0.12 * gait, -0.10, 0.05)
    set_joint("r_ankle", -0.12 * gait, 0.10, 0.05)

    arr = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
    for name, idx in J.items():
        arr[idx] = coords[name]
    return arr


def run_self_tests(H: np.ndarray) -> None:
    """Lightweight in-app checks for geometry/math functions."""
    # Distance symmetry.
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    assert abs(euclidean(a, b) - euclidean(b, a)) < 1e-9

    # Homography forward test.
    pred_img = project_xz_batch(H, FLOOR_METER_PTS)
    assert np.allclose(pred_img, FLOOR_IMG_PTS, atol=1e-1)

    # Homography inverse test.
    Hinv = np.linalg.inv(H)
    back_meter = project_xz_batch(Hinv, pred_img)
    assert np.allclose(back_meter, FLOOR_METER_PTS, atol=1e-2)

    # Zone boundary sanity.
    assert classify_zone(0.40, DEFAULT_CONFIG) == "intimate"
    assert classify_zone(0.80, DEFAULT_CONFIG) == "personal"
    assert classify_zone(2.00, DEFAULT_CONFIG) == "social"
    assert classify_zone(4.00, DEFAULT_CONFIG) == "public"

    logger.info("Self-tests passed: distance symmetry, homography round-trip, zone boundaries.")


def mask_to_segments(mask: np.ndarray, ts: np.ndarray, min_duration_s: float) -> List[Tuple[float, float]]:
    segments = []
    start = None

    for i, val in enumerate(mask):
        if bool(val) and start is None:
            start = float(ts[i])
        elif not bool(val) and start is not None:
            end = float(ts[i])
            if end - start >= min_duration_s:
                segments.append((start, end))
            start = None

    if start is not None:
        end = float(ts[-1]) + 1.0 / float(FPS)
        if end - start >= min_duration_s:
            segments.append((start, end))

    return segments


def build_raw(H: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate deterministic synthetic two-person motion and projected landmarks."""
    rng = np.random.default_rng(SEED)
    n = int(round(DURATION_S * FPS))
    ts = np.linspace(0.0, DURATION_S, n, endpoint=False, dtype=np.float32)

    p1 = np.zeros((n, 2), dtype=np.float32)
    p2 = np.zeros((n, 2), dtype=np.float32)

    phases = rng.uniform(0.0, 2.0 * math.pi, size=4).astype(np.float32)

    for i, t in enumerate(ts):
        t = float(t)

        approach = smoothstep(2.0, 5.0, t)
        separate = smoothstep(7.0, 10.0, t)
        public = smoothstep(10.2, 12.0, t)

        base_x1 = -1.55 + 1.20 * approach - 0.45 * separate - 1.25 * public
        base_x2 =  1.55 - 1.20 * approach + 0.45 * separate + 1.25 * public
        base_z  =  5.65 - 2.55 * approach - 1.35 * separate - 0.35 * public

        x1 = base_x1 + 0.035 * math.sin(2.0 * math.pi * 0.55 * t + float(phases[0]))
        x2 = base_x2 + 0.035 * math.sin(2.0 * math.pi * 0.58 * t + float(phases[1]))
        z1 = base_z  + 0.025 * math.cos(2.0 * math.pi * 0.47 * t + float(phases[2]))
        z2 = base_z  + 0.025 * math.cos(2.0 * math.pi * 0.49 * t + float(phases[3]))

        p1[i] = (x1, z1)
        p2[i] = (x2, z2)

    theta1 = np.zeros(n, dtype=np.float32)
    theta2 = np.zeros(n, dtype=np.float32)

    for i in range(n):
        desired1 = math.atan2(float(p2[i, 0] - p1[i, 0]), float(p2[i, 1] - p1[i, 1]))
        desired2 = math.atan2(float(p1[i, 0] - p2[i, 0]), float(p1[i, 1] - p2[i, 1]))

        if i == 0:
            theta1[i] = desired1
            theta2[i] = desired2
        else:
            theta1[i] = theta1[i - 1] + 0.18 * angle_delta(desired1, float(theta1[i - 1]))
            theta2[i] = theta2[i - 1] + 0.18 * angle_delta(desired2, float(theta2[i - 1]))

            theta1[i] = (float(theta1[i]) + math.pi) % (2.0 * math.pi) - math.pi
            theta2[i] = (float(theta2[i]) + math.pi) % (2.0 * math.pi) - math.pi

    handshake = np.array(
        [
            smoothstep(4.8, 5.1, float(t)) * (1.0 - smoothstep(5.5, 5.8, float(t)))
            for t in ts
        ],
        dtype=np.float32,
    )

    distance = np.zeros(n, dtype=np.float32)
    facing1_deg = np.zeros(n, dtype=np.float32)
    facing2_deg = np.zeros(n, dtype=np.float32)
    wrist_px_min = np.zeros(n, dtype=np.float32)
    shoulder_width_px = np.zeros(n, dtype=np.float32)

    joints1 = np.zeros((n, len(JOINT_NAMES), 3), dtype=np.float32)
    joints2 = np.zeros((n, len(JOINT_NAMES), 3), dtype=np.float32)
    uv1 = np.zeros((n, len(JOINT_NAMES), 2), dtype=np.int32)
    uv2 = np.zeros((n, len(JOINT_NAMES), 2), dtype=np.int32)

    for i, t in enumerate(ts):
        t = float(t)

        distance[i] = euclidean(p1[i], p2[i])

        line1 = math.atan2(float(p2[i, 0] - p1[i, 0]), float(p2[i, 1] - p1[i, 1]))
        line2 = math.atan2(float(p1[i, 0] - p2[i, 0]), float(p1[i, 1] - p2[i, 1]))

        facing1_deg[i] = math.degrees(abs(angle_delta(float(theta1[i]), line1)))
        facing2_deg[i] = math.degrees(abs(angle_delta(float(theta2[i]), line2)))

        reach = float(handshake[i]) * max(0.0, 0.5 * float(distance[i]) - 0.06)

        j1 = body_joints_world(float(p1[i, 0]), float(p1[i, 1]), float(theta1[i]), t, float(handshake[i]), reach)
        j2 = body_joints_world(float(p2[i, 0]), float(p2[i, 1]), float(theta2[i]), t, float(handshake[i]), reach)

        joints1[i] = j1
        joints2[i] = j2

        u1 = project_joint_array(H, j1)
        u2 = project_joint_array(H, j2)

        uv1[i] = u1
        uv2[i] = u2

        sw1 = euclidean(u1[J["l_shoulder"]], u1[J["r_shoulder"]])
        sw2 = euclidean(u2[J["l_shoulder"]], u2[J["r_shoulder"]])
        shoulder_width_px[i] = 0.5 * (sw1 + sw2)

        wrists1 = [u1[J["l_wrist"]], u1[J["r_wrist"]]]
        wrists2 = [u2[J["l_wrist"]], u2[J["r_wrist"]]]

        wrist_px_min[i] = min(
            euclidean(a, b)
            for a in wrists1
            for b in wrists2
        )

    id_confidence = np.ones(n, dtype=np.float32)

    return {
        "H": H,
        "t": ts,
        "p1": p1,
        "p2": p2,
        "theta1": theta1,
        "theta2": theta2,
        "distance": distance,
        "facing1_deg": facing1_deg,
        "facing2_deg": facing2_deg,
        "wrist_px_min": wrist_px_min,
        "shoulder_width_px": shoulder_width_px,
        "handshake_intensity": handshake,
        "joints1": joints1,
        "joints2": joints2,
        "uv1": uv1,
        "uv2": uv2,
        "id_confidence": id_confidence,
    }


def detect_events(raw: Dict[str, np.ndarray], cfg: Dict[str, float]) -> pd.DataFrame:
    ts = raw["t"]
    dist = raw["distance"]
    speed = np.gradient(dist, ts)

    facing1 = raw["facing1_deg"]
    facing2 = raw["facing2_deg"]
    wrist = raw["wrist_px_min"]
    shoulder = np.maximum(raw["shoulder_width_px"], 1.0)
    id_conf = raw["id_confidence"]

    handshake_threshold_px = 0.25 * shoulder

    handshake_mask = (
        (wrist < handshake_threshold_px)
        & (dist < cfg["personal_m"])
        & (id_conf > 0.5)
    )

    approach_mask = (
        (speed < -0.12)
        & (dist > cfg["intimate_m"])
        & (~handshake_mask)
    )

    engaged_mask = (
        (dist < cfg["personal_m"])
        & (facing1 < 45.0)
        & (facing2 < 45.0)
        & (~handshake_mask)
        & (id_conf > 0.5)
    )

    separate_mask = (
        (speed > 0.12)
        & (dist > cfg["personal_m"])
    )

    rows = []

    def add_events(
        event_type: str,
        mask: np.ndarray,
        min_duration_s: float,
        score: np.ndarray,
    ) -> None:
        for start_s, end_s in mask_to_segments(mask, ts, min_duration_s):
            idx = (ts >= start_s) & (ts < end_s)
            if not np.any(idx):
                continue
            conf = float(np.clip(np.nanmean(score[idx]), 0.0, 1.0))
            rows.append(
                {
                    "start_s": round(float(start_s), 3),
                    "end_s": round(float(end_s), 3),
                    "type": event_type,
                    "confidence": round(conf, 3),
                }
            )

    add_events(
        "APPROACH",
        approach_mask,
        0.5,
        np.clip(-speed / 1.0, 0.0, 1.0),
    )

    add_events(
        "HANDSHAKE",
        handshake_mask,
        0.5,
        np.clip(1.0 - wrist / (handshake_threshold_px + 1e-6), 0.0, 1.0),
    )

    engaged_score = np.clip(
        (1.0 - dist / max(cfg["personal_m"], 1e-6))
        * (1.0 - np.minimum(facing1 + facing2, 90.0) / 90.0),
        0.0,
        1.0,
    )
    add_events(
        "ENGAGED",
        engaged_mask,
        1.0,
        engaged_score,
    )

    add_events(
        "SEPARATE",
        separate_mask,
        0.5,
        np.clip(speed / 1.0, 0.0, 1.0),
    )

    events = pd.DataFrame(rows, columns=["start_s", "end_s", "type", "confidence"])
    if not events.empty:
        events = events.sort_values(["start_s", "type"]).reset_index(drop=True)

    return events


def compute_metrics(raw: Dict[str, np.ndarray], cfg: Dict[str, float]) -> Dict[str, object]:
    ts = raw["t"]
    dist = raw["distance"]

    zones = np.array([classify_zone(float(d), cfg) for d in dist], dtype=object)

    pairs = pd.DataFrame(
        {
            "t_s": ts.astype(float),
            "x1": raw["p1"][:, 0].astype(float),
            "z1": raw["p1"][:, 1].astype(float),
            "x2": raw["p2"][:, 0].astype(float),
            "z2": raw["p2"][:, 1].astype(float),
            "distance_m": dist.astype(float),
            "zone": zones,
            "facing1_deg": raw["facing1_deg"].astype(float),
            "facing2_deg": raw["facing2_deg"].astype(float),
        }
    )

    events = detect_events(raw, cfg)

    zone_times = {
        zone: float(np.sum(zones == zone) / float(FPS))
        for zone in ["intimate", "personal", "social", "public"]
    }

    if events.empty:
        event_counts = {}
    else:
        event_counts = {
            str(k): int(v)
            for k, v in events["type"].value_counts().items()
        }

    summary = {
        "mean_distance_m": float(np.mean(dist)),
        "time_in_each_zone_s": zone_times,
        "event_counts": event_counts,
    }

    speed = np.gradient(dist, ts)

    return {
        "pairs": pairs,
        "events": events,
        "summary": summary,
        "zones": zones,
        "speed": speed,
    }


def pt_tuple(p) -> Tuple[int, int]:
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def active_event_types(events_df: pd.DataFrame, t_s: float) -> List[str]:
    if events_df.empty:
        return []
    mask = (events_df["start_s"] <= t_s) & (events_df["end_s"] > t_s)
    return events_df.loc[mask, "type"].tolist()


def draw_camera_frame(
    i: int,
    raw: Dict[str, np.ndarray],
    cfg: Dict[str, float],
    metrics: Dict[str, object],
    show_skeleton: bool = True,
    show_event_overlay: bool = True,
) -> np.ndarray:
    H = raw["H"]
    frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    frame[:] = (45, 40, 35)

    floor_corners = project_xz_batch(
        H,
        np.array(
            [
                [-2.95, 0.05],
                [ 2.95, 0.05],
                [ 2.95, 7.95],
                [-2.95, 7.95],
            ],
            dtype=np.float32,
        ),
    ).astype(np.int32)

    cv2.fillPoly(frame, [floor_corners], (92, 88, 82))

    # Metric floor grid.
    for x in np.arange(-2.5, 2.51, 0.5):
        p0 = project_xz(H, float(x), 0.15)
        p1 = project_xz(H, float(x), 7.85)
        major = abs(float(x) % 1.0) < 1e-6
        color = (125, 120, 115) if major else (78, 75, 72)
        cv2.line(frame, pt_tuple(p0), pt_tuple(p1), color, 1, cv2.LINE_AA)

    for z in np.arange(0.5, 7.51, 0.5):
        p0 = project_xz(H, -2.85, float(z))
        p1 = project_xz(H, 2.85, float(z))
        major = abs(float(z) % 1.0) < 1e-6
        color = (125, 120, 115) if major else (78, 75, 72)
        cv2.line(frame, pt_tuple(p0), pt_tuple(p1), color, 1, cv2.LINE_AA)

    colors = {
        1: (255, 190, 40),   # BGR cyan-ish
        2: (40, 140, 255),   # BGR orange-ish
    }

    # Draw farther person first.
    order = sorted([1, 2], key=lambda pid: float(raw[f"p{pid}"][i, 1]), reverse=True)

    for pid in order:
        uv = raw[f"uv{pid}"][i]
        color = colors[pid]

        if show_skeleton:
            for a, b in BONES:
                cv2.line(frame, pt_tuple(uv[a]), pt_tuple(uv[b]), color, 2, cv2.LINE_AA)

            for p in uv:
                cv2.circle(frame, pt_tuple(p), 2, (240, 240, 240), -1, cv2.LINE_AA)

        ankle_mid = (uv[J["l_ankle"]] + uv[J["r_ankle"]]) // 2
        head = uv[J["head"]]
        shoulder_w = euclidean(uv[J["l_shoulder"]], uv[J["r_shoulder"]])
        box_w = max(18.0, shoulder_w * 1.45)

        x1 = int(ankle_mid[0] - box_w / 2.0)
        x2 = int(ankle_mid[0] + box_w / 2.0)
        y1 = int(head[1] - 8.0)
        y2 = int(ankle_mid[1] + 6.0)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(CAM_W - 1, x2)
        y2 = min(CAM_H - 1, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"ID {pid}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    t = float(raw["t"][i])
    d = float(raw["distance"][i])
    zone = str(metrics["zones"][i])
    active = active_event_types(metrics["events"], t) if show_event_overlay else []

    cv2.rectangle(frame, (0, 0), (CAM_W, 58), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"camera view | t={t:0.2f}s | distance={d:0.2f} m | zone={zone}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    event_text = ", ".join(active) if active else "none"
    cv2.putText(
        frame,
        f"events: {event_text} | ID confidence: 1.00 synthetic",
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_topdown_frame(
    i: int,
    raw: Dict[str, np.ndarray],
    cfg: Dict[str, float],
    metrics: Dict[str, object],
    show_trails: bool = True,
) -> np.ndarray:
    img = np.full((TD_H, TD_W, 3), 22, dtype=np.uint8)

    xmin, xmax = -3.0, 3.0
    zmin, zmax = 0.0, 8.0

    def w2i(x: float, z: float) -> Tuple[int, int]:
        u = (float(x) - xmin) / (xmax - xmin) * float(TD_W - 1)
        v = float(TD_H - 1) - (float(z) - zmin) / (zmax - zmin) * float(TD_H - 1)
        u = int(max(0, min(TD_W - 1, round(u))))
        v = int(max(0, min(TD_H - 1, round(v))))
        return u, v

    # Grid.
    for x in np.arange(-2.5, 2.51, 0.5):
        p0 = w2i(float(x), zmin + 0.1)
        p1 = w2i(float(x), zmax - 0.1)
        major = abs(float(x) % 1.0) < 1e-6
        color = (105, 105, 105) if major else (55, 55, 55)
        cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)

    for z in np.arange(0.5, 7.51, 0.5):
        p0 = w2i(xmin + 0.1, float(z))
        p1 = w2i(xmax - 0.1, float(z))
        major = abs(float(z) % 1.0) < 1e-6
        color = (105, 105, 105) if major else (55, 55, 55)
        cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)

    # Axis labels.
    for x in range(-2, 3):
        u, v = w2i(float(x), zmin + 0.35)
        cv2.putText(
            img,
            f"{x} m",
            (u - 10, v + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (165, 165, 165),
            1,
            cv2.LINE_AA,
        )

    for z in range(1, 8):
        u, v = w2i(xmin + 0.35, float(z))
        cv2.putText(
            img,
            f"{z} m",
            (u + 4, v + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (165, 165, 165),
            1,
            cv2.LINE_AA,
        )

    colors = {
        1: (255, 190, 40),
        2: (40, 140, 255),
    }

    if show_trails:
        start = max(0, i - 90)
        for pid in [1, 2]:
            pts = [w2i(float(raw[f"p{pid}"][j, 0]), float(raw[f"p{pid}"][j, 1])) for j in range(start, i + 1)]
            if len(pts) < 2:
                continue
            for k in range(len(pts) - 1):
                alpha = 0.25 + 0.75 * (k / max(1, len(pts) - 2))
                col = tuple(int(c * alpha) for c in colors[pid])
                cv2.line(img, pts[k], pts[k + 1], col, 2, cv2.LINE_AA)

    p1_xy = w2i(float(raw["p1"][i, 0]), float(raw["p1"][i, 1]))
    p2_xy = w2i(float(raw["p2"][i, 0]), float(raw["p2"][i, 1]))

    cv2.line(img, p1_xy, p2_xy, (190, 190, 190), 1, cv2.LINE_AA)

    mid = ((p1_xy[0] + p2_xy[0]) // 2, (p1_xy[1] + p2_xy[1]) // 2)
    cv2.putText(
        img,
        f"{float(raw['distance'][i]):.2f} m",
        (mid[0] + 8, mid[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    for pid, theta in ((1, float(raw["theta1"][i])), (2, float(raw["theta2"][i]))):
        x = float(raw[f"p{pid}"][i, 0])
        z = float(raw[f"p{pid}"][i, 1])
        u, v = w2i(x, z)

        fx = math.sin(theta)
        fz = math.cos(theta)
        end = w2i(x + fx * 0.8, z + fz * 0.8)

        cv2.arrowedLine(img, (u, v), end, colors[pid], 2, cv2.LINE_AA, 0, 0.22)
        cv2.circle(img, (u, v), 7, colors[pid], -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"P{pid}",
            (u + 9, v - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colors[pid],
            2,
            cv2.LINE_AA,
        )

    if float(raw["handshake_intensity"][i]) > 0.15:
        for joints in (raw["joints1"][i], raw["joints2"][i]):
            for name in ("l_wrist", "r_wrist"):
                jt = joints[J[name]]
                u, v = w2i(float(jt[0]), float(jt[2]))
                cv2.circle(img, (u, v), 3, (80, 255, 80), -1, cv2.LINE_AA)

    t = float(raw["t"][i])
    zone = str(metrics["zones"][i])
    active = active_event_types(metrics["events"], t)

    cv2.rectangle(img, (0, 0), (TD_W, 42), (0, 0, 0), -1)
    cv2.putText(
        img,
        f"top-down metric map | zone={zone}",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    event_text = ", ".join(active) if active else "none"
    cv2.putText(
        img,
        f"events: {event_text}",
        (8, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    return img


def render_combined_frame(
    i: int,
    raw: Dict[str, np.ndarray],
    cfg: Dict[str, float],
    metrics: Dict[str, object],
    show_skeleton: bool = True,
    show_trails: bool = True,
    show_event_overlay: bool = True,
) -> np.ndarray:
    cam = draw_camera_frame(i, raw, cfg, metrics, show_skeleton=show_skeleton, show_event_overlay=show_event_overlay)
    td = draw_topdown_frame(i, raw, cfg, metrics, show_trails=show_trails)
    return np.hstack([cam, td])


def make_writer(path: str, size: Tuple[int, int]):
    last_name = None
    for name in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*name)
        writer = cv2.VideoWriter(path, fourcc, float(FPS), size)
        if writer.isOpened():
            return writer, name
        writer.release()
        last_name = name
    logger.warning("Could not open VideoWriter with tried codecs: %s", last_name)
    return None, None


def encode_videos(
    raw: Dict[str, np.ndarray],
    cfg: Dict[str, float],
    metrics: Dict[str, object],
) -> Tuple[bytes, bytes]:
    comb_fd, comb_path = tempfile.mkstemp(suffix=".mp4")
    td_fd, td_path = tempfile.mkstemp(suffix=".mp4")
    os.close(comb_fd)
    os.close(td_fd)

    try:
        w_comb, cc_comb = make_writer(comb_path, (CAM_W + TD_W, CAM_H))
        w_td, cc_td = make_writer(td_path, (TD_W, TD_H))

        if w_comb is None or w_td is None:
            raise RuntimeError("OpenCV VideoWriter could not open an MP4 encoder in this environment.")

        n = len(raw["t"])
        for i in range(n):
            td = draw_topdown_frame(i, raw, cfg, metrics, show_trails=True)
            cam = draw_camera_frame(i, raw, cfg, metrics, show_skeleton=True, show_event_overlay=True)
            comb = np.hstack([cam, td])

            w_comb.write(comb)
            w_td.write(td)

        w_comb.release()
        w_td.release()

        with open(comb_path, "rb") as f:
            comb_bytes = f.read()
        with open(td_path, "rb") as f:
            td_bytes = f.read()

        if not comb_bytes or not td_bytes:
            raise RuntimeError("Encoded video bytes are empty.")

        logger.info(
            "Encoded videos | combined_fourcc=%s topdown_fourcc=%s combined_bytes=%d topdown_bytes=%d",
            cc_comb,
            cc_td,
            len(comb_bytes),
            len(td_bytes),
        )

        return comb_bytes, td_bytes

    finally:
        for p in (comb_path, td_path):
            try:
                os.unlink(p)
            except OSError:
                pass


@st.cache_data(show_spinner="Building synthetic PairMotion demo...")
def build_assets() -> Dict[str, object]:
    H = cv2.getPerspectiveTransform(FLOOR_METER_PTS, FLOOR_IMG_PTS)
    run_self_tests(H)

    raw = build_raw(H)
    metrics = compute_metrics(raw, DEFAULT_CONFIG)
    demo_bytes, topdown_bytes = encode_videos(raw, DEFAULT_CONFIG, metrics)

    return {
        "raw": raw,
        "default_metrics": metrics,
        "demo_bytes": demo_bytes,
        "topdown_bytes": topdown_bytes,
        "H": H,
    }


def mp4_html(mp4_bytes: bytes) -> str:
    b64 = base64.b64encode(mp4_bytes).decode("ascii")
    return (
        '<video controls autoplay muted loop playsinline '
        'style="width:100%; border-radius:10px; background:#000;">'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
        "</video>"
    )


def main() -> None:
    st.set_page_config(
        page_title="PairMotion Demo",
        page_icon="🧍♀️🧍‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("PairMotion: Two-Person Interaction Measurement")
    st.caption(
        "Synthetic demo only. No real people, no raw video, no API key, no paid service. "
        "Do not use for surveillance, emotion inference, or medical/safety claims."
    )

    assets = build_assets()
    raw = assets["raw"]

    st.sidebar.title("Controls")

    show_demo_video = st.sidebar.checkbox("Autoplay demo video", value=True)
    show_skeleton = st.sidebar.checkbox("Show skeletons", value=True)
    show_trails = st.sidebar.checkbox("Show top-down trails", value=True)
    show_event_overlay = st.sidebar.checkbox("Show event overlay", value=True)

    st.sidebar.subheader("Proxemic thresholds")
    intimate = st.sidebar.slider("Intimate upper bound (m)", 0.10, 0.90, 0.45, 0.05)
    personal = st.sidebar.slider(
        "Personal upper bound (m)",
        intimate + 0.05,
        2.50,
        max(1.20, intimate + 0.10),
        0.05,
    )
    social = st.sidebar.slider(
        "Social upper bound (m)",
        personal + 0.05,
        6.00,
        max(3.60, personal + 0.20),
        0.10,
    )

    cfg = {
        "intimate_m": float(intimate),
        "personal_m": float(personal),
        "social_m": float(social),
    }

    max_t = float(raw["t"][-1])
    t_sel = st.sidebar.slider(
        "Scrub time (s)",
        0.0,
        max_t,
        0.0,
        step=1.0 / float(FPS),
        format="%.2f",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Privacy/safety note:** this demo uses synthetic motion. "
        "For real deployments, use only consented recordings and keep skeleton data separate from raw video."
    )

    metrics = compute_metrics(raw, cfg)

    if show_demo_video:
        st.subheader("Live synthetic demo video")
        try:
            st.video(
                assets["demo_bytes"],
                format="video/mp4",
                loop=True,
                muted=True,
                autoplay=True,
            )
        except TypeError:
            st.markdown(mp4_html(assets["demo_bytes"]), unsafe_allow_html=True)

    tab_live, tab_tables, tab_outputs, tab_limits = st.tabs(
        ["Live view", "Tables", "Outputs", "Known limits"]
    )

    with tab_live:
        idx = int(np.clip(round(t_sel * FPS), 0, len(raw["t"]) - 1))
        frame = render_combined_frame(
            idx,
            raw,
            cfg,
            metrics,
            show_skeleton=show_skeleton,
            show_trails=show_trails,
            show_event_overlay=show_event_overlay,
        )

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        t = float(raw["t"][idx])
        d = float(raw["distance"][idx])
        zone = str(metrics["zones"][idx])
        active = active_event_types(metrics["events"], t)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current time", f"{t:.2f} s")
        c2.metric("Current distance", f"{d:.2f} m")
        c3.metric("Current zone", zone)
        c4.metric("Active events", ", ".join(active) if active else "none")

        st.subheader("Distance over time")
        chart_df = metrics["pairs"].set_index("t_s")
        st.line_chart(chart_df[["distance_m"]], height=230)

        st.subheader("Facing angle over time")
        st.line_chart(chart_df[["facing1_deg", "facing2_deg"]], height=230)

    with tab_tables:
        st.subheader("pairs.csv")
        st.dataframe(metrics["pairs"], use_container_width=True, height=330)

        st.subheader("events.csv")
        st.dataframe(metrics["events"], use_container_width=True, height=260)

        st.subheader("summary.json")
        st.json(metrics["summary"])

    with tab_outputs:
        st.subheader("Download outputs")

        pairs_csv = metrics["pairs"].to_csv(index=False).encode("utf-8")
        events_csv = metrics["events"].to_csv(index=False).encode("utf-8")
        summary_json = json.dumps(metrics["summary"], indent=2).encode("utf-8")

        st.download_button(
            label="Download pairs.csv",
            data=pairs_csv,
            file_name="pairs.csv",
            mime="text/csv",
        )

        st.download_button(
            label="Download events.csv",
            data=events_csv,
            file_name="events.csv",
            mime="text/csv",
        )

        st.download_button(
            label="Download summary.json",
            data=summary_json,
            file_name="summary.json",
            mime="application/json",
        )

        st.download_button(
            label="Download topdown.mp4",
            data=assets["topdown_bytes"],
            file_name="topdown.mp4",
            mime="video/mp4",
        )

        st.download_button(
            label="Download demo.mp4",
            data=assets["demo_bytes"],
            file_name="demo.mp4",
            mime="video/mp4",
        )

        if st.button("Save outputs to ./outputs"):
            out_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(out_dir, exist_ok=True)

            metrics["pairs"].to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
            metrics["events"].to_csv(os.path.join(out_dir, "events.csv"), index=False)

            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(metrics["summary"], f, indent=2)

            with open(os.path.join(out_dir, "topdown.mp4"), "wb") as f:
                f.write(assets["topdown_bytes"])

            with open(os.path.join(out_dir, "demo.mp4"), "wb") as f:
                f.write(assets["demo_bytes"])

            st.success(f"Saved outputs to: `{out_dir}`")

    with tab_limits:
        st.markdown(
            """
            ### Known limits

            - This single-file demo uses **synthetic motion**, not real pose estimation.
            - The camera view is a heuristic projection from a floor homography, not a calibrated metric 3D reconstruction.
            - Identities are stable by construction in the synthetic demo; real ID-switch prevention requires tracking logic.
            - Handshake detection here uses projected wrist pixel distance and shoulder-width heuristics.
            - No medical, safety, emotion, or relationship inference is performed.
            - For real video, use only consented recordings and keep skeleton data separate from raw video.
            """
        )


if __name__ == "__main__":
    main()
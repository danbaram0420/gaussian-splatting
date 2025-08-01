import json, numpy as np
from scene.cameras import Camera
from utils.graphics_utils import fov2focal
from pytorch3d.transforms import quaternion_to_matrix
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch, cv2, PIL.Image as PIL


def _split_RT(mat, world_to_cam=True):
    """4×4 행렬을 R(3×3), T(3,) 로 분리.
       world_to_cam=True  → W2C 행렬이므로 R, T 그대로 반환
       world_to_cam=False → C2W 행렬이므로 역행렬로 변환"""
    mat = np.array(mat, np.float32).reshape(4, 4)
    if not world_to_cam:          # c2w → w2c
        mat = np.linalg.inv(mat)
    R = mat[:3, :3]
    T = mat[:3, 3]
    return R, T

def _dummy_pil(w, h):            # 검정 더미 이미지
    return PIL.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

def _cam_from_graphdeco(rec, uid):
    W, H = rec["width"], rec["height"]
    fx, fy = rec["fx"], rec["fy"]

    # ── R, T 파싱 ────────────────────────────────────────────────
    if "rotation" in rec:                                    # (a) 별도 3×3
        R = np.array(rec["rotation"], np.float32)
        if "position" in rec:                                #    + position
            T = np.array(rec["position"], np.float32)
        elif all(k in rec for k in ("tx", "ty", "tz")):      #    + tx,ty,tz
            T = np.array([rec["tx"], rec["ty"], rec["tz"]], np.float32)
        else:
            raise KeyError("No translation in record.")
    elif "transform" in rec:                                 # (b) transform 4×4 (C2W)
        R, T = _split_RT(rec["transform"], world_to_cam=False)
    elif "c2w" in rec:                                       # (c) c2w 4×4
        R, T = _split_RT(rec["c2w"], world_to_cam=False)
    elif "w2c" in rec:                                       # (d) w2c 4×4
        R, T = _split_RT(rec["w2c"], world_to_cam=True)
    else:                                                    # (e) 쿼터니언 + tx
        quat = np.array([rec["qw"], rec["qx"], rec["qy"], rec["qz"]], np.float32)
        R = quaternion_to_matrix(torch.from_numpy(quat)).numpy()
        T = np.array([rec["tx"], rec["ty"], rec["tz"]], np.float32)

    # FoV
    FoVx = 2 * np.arctan(W / (2 * fx))
    FoVy = 2 * np.arctan(H / (2 * fy))

    return Camera(
        (W, H),          # resolution
        uid,             # colmap_id
        R, T,
        np.rad2deg(FoVx), np.rad2deg(FoVy),
        None,            # depth_params
        _dummy_pil(W, H),
        None,            # invdepthmap
        f"cam_{uid:04d}.png",
        uid
    )

def load_cameras(path:str):
    """transforms.json  /  GraphDeco cameras.json  /  COLMAP 폴더 지원"""
    if path.endswith(".json"):
        with open(path) as f:
            meta = json.load(f)
        # Nerf-synthetic (dict+frames) ↔ GraphDeco (list[])
        if isinstance(meta, dict) and "frames" in meta:                # Nerf-synthetic
            W, H = meta["w"], meta["h"]
            fx = fy = 0.5 * W / np.tan(meta["camera_angle_x"] / 2)
            cams = []
            for uid, fr in enumerate(meta["frames"]):
                R, T = _split_RT(fr["transform_matrix"], False)
                cams.append(_cam_from_graphdeco({
                    "width": W, "height": H,
                    "fx": fx, "fy": fy,
                    "rotation": R.tolist(), "position": T.tolist()
                }, uid))
            return cams
        elif isinstance(meta, list):                                  # GraphDeco list
            return [_cam_from_graphdeco(rec, uid) for uid, rec in enumerate(meta)]
        else:
            raise ValueError("Unknown JSON camera format.")
    else:
        # --- COLMAP 폴더는 필요시 여기에 구현 ---
        raise NotImplementedError("COLMAP txt parsing not yet implemented.")
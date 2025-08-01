"""
orient_normals_from_ply.py
Vanilla 3-D Gaussian-Splatting용 – 외향 노멀 일괄 정렬 스크립트
"""
import argparse, torch
from tqdm import tqdm

from scene.gaussian_model import GaussianModel         # 노멀 버퍼 포함
from gaussian_renderer import render# 기존 렌더러
from arguments import PipelineParams
from utils.camera_io import load_cameras                      # 사용자 제공 util

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply_path",  required=True, help="point_cloud.ply (vanilla 3-DGS)")
    p.add_argument("--cam_file",  required=True, help="transforms.json -or- cameras.txt 폴더")
    p.add_argument("--out_ply",   default="point_cloud_oriented.ply")
    p.add_argument("--white_background", action="store_true")
    p.add_argument("--sh_degree", type=int, default=3)
    args = p.parse_args()

    # ① 가우시안 로드
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(args.ply_path)

    # ② 카메라 로드
    cameras   = load_cameras(args.cam_file)                # list[Camera]

    # ③ 배경색 & 파이프라인 설정
    bg = torch.tensor([1,1,1] if args.white_background else [0,0,0],
                      dtype=torch.float32, device="cuda")
    pipe = PipelineParams(argparse.ArgumentParser())                            # 기본값이면 충분

    # ④ 각 뷰를 한 번씩 렌더 → 노멀 투표 누적
    for cam in tqdm(cameras, desc="normal-vote"):
        render(cam, gaussians, pipe, bg, correct_norml=True)

    # ⑤ 최종 PLY 저장(노멀 포함)
    gaussians.save_ply(args.out_ply, orient_normals=True)
    print(f"[✓]  Oriented PLY saved to {args.out_ply}")

if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
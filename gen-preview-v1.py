import cv2
import numpy as np
import random
import os

def extract_frames_with_chromakey(video_path, output_dir, frame_count=4, tolerance=30, feather_ratio=0.005):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < frame_count:
        raise ValueError("视频帧数不足")

    # 随机挑选帧
    step = total_frames // (frame_count + 1)
    candidate_frames = list(range(step, total_frames, step))
    selected_frames = random.sample(candidate_frames, frame_count)

    os.makedirs(output_dir, exist_ok=True)

    feather = max(1, int(width * feather_ratio))

    for i, frame_no in enumerate(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # ==== 生成绿幕 mask ====
        lower = np.array([0, 255 - tolerance, 0])
        upper = np.array([tolerance, 255, tolerance])
        mask = cv2.inRange(frame, lower, upper)

        # ==== alpha 羽化 ====
        mask_blur = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
        alpha = 255 - mask_blur
        alpha_f = alpha.astype(np.float32) / 255.0

        # ==== 去绿边（despill）====
        fg = frame.astype(np.float32)
        R, G, B = fg[:, :, 2], fg[:, :, 1], fg[:, :, 0]
        # 压制绿色分量
        G = np.minimum(G, (R + B) / 2)
        fg[:, :, 1] = G

        # 应用 alpha
        for c in range(3):
            fg[:, :, c] = fg[:, :, c] * alpha_f

        # 合并 alpha
        frame_bgra[:, :, :3] = fg.astype(np.uint8)
        frame_bgra[:, :, 3] = (alpha_f * 255).astype(np.uint8)

        out_path = os.path.join(output_dir, f"frame_{i+1}.png")
        cv2.imwrite(out_path, frame_bgra)
        print(f"已保存: {out_path}")

    cap.release()

if __name__ == "__main__":
    video_file = "A.4376.AI绿幕.mp4"
    output_folder = "output_frames"
    extract_frames_with_chromakey(video_file, output_folder)
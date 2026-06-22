import cv2
import numpy as np
import subprocess
import os

def video_to_transparent_mov(video_path, output_path, tolerance=30, feather_ratio=0.005,
                             max_frames=100, frame_step=2, speed_factor=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    feather = max(1, int(width * feather_ratio))
    frames = []

    frame_idx = 0
    saved_frames = 0

    print(f"视频总帧数: {total_frames}, 帧率: {fps}, 尺寸: {width}x{height}")
    print(f"开始处理视频，每 {frame_step} 帧抽取一帧，最多保存 {max_frames} 帧")

    while True:
        ret, frame = cap.read()
        if not ret or saved_frames >= max_frames:
            break

        if frame_idx % frame_step == 0:
            # ==== 生成绿幕 mask ====
            lower = np.array([0, 255 - tolerance, 0])
            upper = np.array([tolerance, 255, tolerance])
            mask = cv2.inRange(frame, lower, upper)

            # ==== alpha 羽化 ====
            mask_blur = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
            alpha = 255 - mask_blur
            alpha_f = alpha.astype(np.float32) / 255.0

            # ==== 去绿边 ====
            fg = frame.astype(np.float32)
            R, G, B = fg[:, :, 2], fg[:, :, 1], fg[:, :, 0]
            G = np.minimum(G, (R + B) / 2)
            fg[:, :, 1] = G

            # ⚠️ 不做预乘，保持颜色正常
            fg = fg.astype(np.uint8)
            alpha_channel = (alpha_f * 255).astype(np.uint8)

            # ==== 直接拼接成 BGRA ====
            bgra = cv2.merge([fg[:, :, 0], fg[:, :, 1], fg[:, :, 2], alpha_channel])

            frames.append(bgra)
            saved_frames += 1

            if saved_frames % 10 == 0 or saved_frames == max_frames:
                print(f"已处理帧: {saved_frames}/{max_frames}")

        frame_idx += 1

    cap.release()

    if not frames:
        print("没有生成任何帧，MOV为空")
        return

    print(f"共生成帧: {len(frames)}")

    # ==== 临时保存 PNG 序列 ====
    tmp_dir = "tmp_frames"
    os.makedirs(tmp_dir, exist_ok=True)

    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i:04d}.png"), f)

    # ==== 用 ffmpeg 合成透明 MOV (ProRes 4444) ====
    fps_out = fps / frame_step * speed_factor
    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps_out}",
        "-i", os.path.join(tmp_dir, "frame_%04d.png"),
        "-c:v", "prores_ks",
        "-pix_fmt", "yuva444p10le",   # 保留透明通道
        output_path
    ]
    subprocess.run(cmd)

    print(f"透明 MOV 已保存: {output_path}")


if __name__ == "__main__":
    video_file = "A.4376.AI绿幕.mp4"
    output_mov  = "output_chroma.mov"
    video_to_transparent_mov(
        video_file,
        output_mov,
        tolerance=30,
        feather_ratio=0.005,
        max_frames=60,
        frame_step=10,
        speed_factor=3.0
    )
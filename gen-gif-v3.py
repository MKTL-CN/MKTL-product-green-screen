import cv2
import numpy as np
import imageio

def video_to_transparent_gif(video_path, output_path, tolerance=30, feather_ratio=0.005,
                             max_frames=100, frame_step=2, palettesize=128, speed_factor=2.0):
    """
    speed_factor: 播放速度倍数，>1加快，<1减慢
    """
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

            # ==== 前景预乘 alpha ====
            for c in range(3):
                fg[:, :, c] = fg[:, :, c] * alpha_f

            # 转 RGBA
            rgba = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = (alpha_f * 255).astype(np.uint8)  # 保留透明度

            frames.append(rgba)
            saved_frames += 1

            if saved_frames % 10 == 0 or saved_frames == max_frames:
                print(f"已处理帧: {saved_frames}/{max_frames}")

        frame_idx += 1

    cap.release()

    if not frames:
        print("没有生成任何帧，GIF为空")
        return

    # ==== 计算每帧显示时间，避免速度不稳定 ====
    duration_per_frame = 1 / (fps / frame_step * speed_factor)  # 秒/帧

    # ==== 保存透明 GIF ====
    imageio.mimsave(
        output_path,
        frames,
        format='GIF',
        loop=0,
        duration=duration_per_frame,
        palettesize=palettesize,
        disposal=2
    )

    print(f"透明 GIF 已保存: {output_path}, 播放速度 x{speed_factor}")


if __name__ == "__main__":
    video_file = "A.4376.AI绿幕.mp4"
    output_gif  = "output_chroma.gif"
    video_to_transparent_gif(
        video_file,
        output_gif,
        tolerance=30,
        feather_ratio=0.005,
        max_frames=60,
        frame_step=10,
        palettesize=128,
        speed_factor=3.0
    )
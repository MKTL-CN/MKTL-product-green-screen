import cv2
import numpy as np
import imageio

def video_to_transparent_gif(video_path, output_path, tolerance=30, feather_ratio=0.005, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    feather = max(1, int(width * feather_ratio))
    frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        frame_idx += 1

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

        # 应用 alpha
        for c in range(3):
            fg[:, :, c] = fg[:, :, c] * alpha_f

        # 转 RGBA
        rgba = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = (alpha_f * 255).astype(np.uint8)

        # 转 ImageIO 可用格式 (uint8, RGBA)
        frames.append(rgba)

        if frame_idx % 50 == 0:
            print(f"处理帧: {frame_idx}/{total_frames}")

    cap.release()

    # ==== 保存透明 GIF ====
    # disposal=2 保证上一帧透明区域不会被填黑
    imageio.mimsave(output_path, frames, fps=int(fps), loop=0, disposal=2, palettesize=256)
    print(f"透明 GIF 已保存: {output_path}")


if __name__ == "__main__":
    video_file = "A.4376.AI绿幕.mp4"
    output_gif  = "output_chroma.gif"
    video_to_transparent_gif(video_file, output_gif)
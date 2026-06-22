import cv2
import numpy as np

def png_to_transparent_png(input_path, output_path,
                           lower_h=40, upper_h=80,
                           min_s=80, min_v=80,
                           feather=3,
                           save_preview=True):
    # 读取图片（BGR）
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print("无法读取图片:", input_path)
        return

    # 转 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义绿色范围
    lower_green = np.array([lower_h, min_s, min_v])
    upper_green = np.array([upper_h, 255, 255])

    # 生成 mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 反向 alpha（255 = 不透明）
    alpha = 255 - mask

    # 羽化边缘（确保核为奇数且 >=1）
    if feather > 0:
        k = max(1, feather*2+1)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    # ==== 去绿边 ====
    fg = img.astype(np.float32)  # BGR float
    R, G, B = fg[:, :, 2], fg[:, :, 1], fg[:, :, 0]
    G = np.minimum(G, (R + B) / 2)
    fg[:, :, 1] = G

    # ==== 应用 alpha（预乘） ====
    alpha_f = alpha.astype(np.float32) / 255.0
    for c in range(3):
        fg[:, :, c] *= alpha_f

    # ==== 准备保存：OpenCV 要求 BGRA 顺序，所以手动 merge 成 BGRA ====
    fg_uint8 = np.clip(fg, 0, 255).astype(np.uint8)
    alpha_uint8 = np.clip(alpha, 0, 255).astype(np.uint8)
    # 注意 merge 顺序是 B, G, R, A（不是 R, G, B, A）
    bgra = cv2.merge([fg_uint8[:, :, 0], fg_uint8[:, :, 1], fg_uint8[:, :, 2], alpha_uint8])

    # 保存透明 PNG（BGRA）
    success = cv2.imwrite(output_path, bgra)
    if success:
        print(f"透明 PNG 已保存: {output_path}")
    else:
        print("保存失败:", output_path)

    # 生成预览（白底合成）并保存（BGR）
    if save_preview:
        # bg 白色浮点
        bg = np.ones_like(img, dtype=np.float32) * 255.0
        # fg 是已预乘的 float，alpha_f 用于合成
        preview_f = fg + bg * (1.0 - alpha_f[..., None])
        preview = np.clip(preview_f, 0, 255).astype(np.uint8)
        preview_file = output_path.replace(".png", "_bg.png")
        cv2.imwrite(preview_file, preview)
        print(f"预览 PNG 已保存: {preview_file}")


if __name__ == "__main__":
    while True:
        input_png  = input("Image Path: ").strip()
        if not input_png:
            break
        output_png = input_png.replace("-green","")
        png_to_transparent_png(input_png, output_png,
                            lower_h=45, upper_h=75,
                            min_s=100, min_v=100,
                            feather=3,
                            save_preview=True)
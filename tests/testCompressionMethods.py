import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_pair(out_path, a, b, title_left, title_right):
    a = np.asarray(a)
    b = np.asarray(b)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(a, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title(title_left)
    ax[0].axis("off")
    ax[1].imshow(b, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title(title_right)
    ax[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    # Ensure repo root is on sys.path so `import methods.*` works even when
    # running this file directly (sys.path[0] would otherwise be ./tests).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import methods.compressionMethods as cm
    import methods.invCompressionMethods as inv

    img_path = os.path.join(repo_root, "test_image.jpg")
    out_dir = os.path.join(repo_root, "out")
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(img_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = cm.convert_to_grayscale(arr)
    h, w = gray.shape

    save_pair(os.path.join(out_dir, "original.png"), gray, gray, "original", "original")

    for k in (8, 16, 32):
        c = cm.dct_k(gray, k)
        r = inv.idct_k(c, (h, w))
        r = np.clip(r, 0.0, 1.0)
        save_pair(
            os.path.join(out_dir, f"dct_k{k}.png"),
            gray,
            r,
            "original",
            f"dct_k{k}",
        )

    for k in (8, 16, 32):
        c = cm.dft_k(gray, k, norm="ortho")
        r = inv.idft_k(c, (h, w), output="real", norm="ortho")
        r = np.clip(r, 0.0, 1.0)
        save_pair(
            os.path.join(out_dir, f"dft_k{k}.png"),
            gray,
            r,
            "original",
            f"dft_k{k}",
        )

    # Wavelet (new API): forward returns (coeff_array, meta) and inverse consumes both.
    for levels in (2, 3, 4):
        for keep in (0, 1, levels):
            coeff_arr, meta = cm.dwt_keep_scales_fwd(gray, levels=levels, keep_levels=keep)
            r = inv.dwt_keep_scales_inv(coeff_arr, meta, output="real")
            r = np.clip(r, 0.0, 1.0)
            save_pair(
                os.path.join(out_dir, f"dwt_keep_scales_L{levels}_K{keep}.png"),
                gray,
                r,
                "original",
                f"dwt_keep_scales_L{levels}_K{keep}",
            )

    try:
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        y = cm.conv2d(gray, k)
        y = np.clip(y, 0.0, 1.0)
        save_pair(
            os.path.join(out_dir, "conv2d_laplacian.png"),
            gray,
            y,
            "original",
            "conv2d_laplacian",
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()



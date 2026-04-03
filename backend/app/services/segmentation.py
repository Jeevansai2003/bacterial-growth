import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import io, color, filters, transform, morphology, segmentation, feature, measure
from skimage.filters import gaussian
from scipy import ndimage as ndi


SUPPORTED_METHODS = {
    "Otsu",
    "Adaptive",
    "Manual",
    "Majority Fusion",
    "CW-MTF (Novel)",
}


# ---------------- COLOR CLUSTERING ----------------
def _kmeans_hsv_cluster(hsv_image, k=3, sample_cap=50000, sat_min=0.35, val_min=0.25):
    H, S, V = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]
    mask = (S >= sat_min) & (V >= val_min)
    pts = np.stack([H[mask], S[mask], V[mask]], axis=1)

    if pts.shape[0] < 200:
        return None, None, None

    if pts.shape[0] > sample_cap:
        idx = np.random.choice(pts.shape[0], size=sample_cap, replace=False)
        pts = pts[idx]

    Z = pts.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers, labels.flatten(), pts


def get_color_likelihood_mask(hsv_image):
    target_hue = 0.13

    res = _kmeans_hsv_cluster(hsv_image, k=3)
    if res[0] is None:
        lower_yellow = (0.10, 0.4, 0.4)
        upper_yellow = (0.18, 1.0, 1.0)
        H, S, V = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]

        hard = (
            (H >= lower_yellow[0]) & (H <= upper_yellow[0]) &
            (S >= lower_yellow[1]) & (V >= lower_yellow[2])
        )
        return gaussian(hard.astype(np.float32), sigma=1.0)

    centers, _, _ = res
    hue_dist = np.abs(centers[:, 0] - target_hue)

    penalty = (
        (centers[:, 1] < 0.25).astype(np.float32) * 0.25 +
        (centers[:, 2] < 0.25).astype(np.float32) * 0.25
    )

    chosen = int(np.argmin(hue_dist + penalty))
    c = centers[chosen]

    H, S, V = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]

    dh = np.minimum(np.abs(H - c[0]), 1.0 - np.abs(H - c[0]))
    ds = S - c[1]
    dv = V - c[2]

    sh, ss, sv = 0.06, 0.25, 0.25
    d2 = (dh / sh) ** 2 + (ds / ss) ** 2 + (dv / sv) ** 2

    likelihood = np.exp(-0.5 * d2).astype(np.float32)
    likelihood = gaussian(likelihood, sigma=1.0)

    return np.clip(likelihood, 0.0, 1.0)


# ---------------- CONFIDENCE FUNCTIONS ----------------
def otsu_confidence(img01):
    t = filters.threshold_otsu(img01)

    hist, bins = np.histogram(img01.ravel(), bins=256, range=(0, 1), density=True)
    mids = (bins[:-1] + bins[1:]) / 2

    left = mids <= t
    w0 = hist[left].sum()
    w1 = hist[~left].sum()

    if w0 < 1e-8 or w1 < 1e-8:
        return 0.0

    mu0 = (hist[left] * mids[left]).sum() / w0
    mu1 = (hist[~left] * mids[~left]).sum() / w1
    muT = (hist * mids).sum()

    sigma_b2 = w0 * (mu0 - muT) ** 2 + w1 * (mu1 - muT) ** 2
    sigma_t2 = ((img01 - img01.mean()) ** 2).mean() + 1e-8

    return float(np.clip(sigma_b2 / sigma_t2, 0, 1))


def adaptive_confidence(img01, block=15):
    local_mean = gaussian(img01, sigma=block / 6.0)
    local_var = gaussian((img01 - local_mean) ** 2, sigma=1.0).mean()
    return float(np.clip(1.0 / (1.0 + 10.0 * local_var), 0, 1))


def manual_confidence(img01, t):
    hist, _ = np.histogram(img01.ravel(), bins=256, range=(0, 1), density=True)

    idx = int(np.clip(t * 255, 1, 254))
    valley = (hist[idx - 1] + hist[idx] + hist[idx + 1]) / 3.0
    peak = hist.max() + 1e-8

    return float(np.clip(1.0 - valley / peak, 0, 1))


# ---------------- CW-MTF ----------------
def cw_mtf_fusion(preprocessed01, manual_t, color_likelihood01):
    t_otsu = filters.threshold_otsu(preprocessed01)
    otsu_mask = preprocessed01 > t_otsu

    adaptive_th = filters.threshold_local(preprocessed01, block_size=15)
    adaptive_mask = preprocessed01 > adaptive_th

    manual_mask = preprocessed01 > manual_t

    wo = otsu_confidence(preprocessed01)
    wa = adaptive_confidence(preprocessed01)
    wm = manual_confidence(preprocessed01, manual_t)
    wc = float(np.clip(color_likelihood01.mean(), 0, 1)) * 0.75

    wsum = wo + wa + wm + wc + 1e-8
    wo, wa, wm, wc = wo / wsum, wa / wsum, wm / wsum, wc / wsum

    P = (
        wo * otsu_mask.astype(np.float32) +
        wa * adaptive_mask.astype(np.float32) +
        wm * manual_mask.astype(np.float32) +
        wc * (color_likelihood01 > 0.5).astype(np.float32)
    )

    fusion_mask = P >= 0.5

    uncertainty = 1.0 - (np.abs(P - 0.5) / 0.5)
    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)

    return fusion_mask, uncertainty, (wo, wa, wm, wc)


# ---------------- METHOD SELECTOR ----------------
def get_binary_mask(preprocessed_image, color_likelihood01, manual_threshold, method):
    if method == "Otsu":
        return preprocessed_image > filters.threshold_otsu(preprocessed_image), None, None

    if method == "Adaptive":
        return preprocessed_image > filters.threshold_local(preprocessed_image, block_size=15), None, None

    if method == "Manual":
        return preprocessed_image > manual_threshold, None, None

    if method == "Majority Fusion":
        otsu = preprocessed_image > filters.threshold_otsu(preprocessed_image)
        adaptive = preprocessed_image > filters.threshold_local(preprocessed_image, block_size=15)
        manual = preprocessed_image > manual_threshold

        fusion = (otsu.astype(np.uint8) + adaptive.astype(np.uint8) + manual.astype(np.uint8)) >= 2
        #fusion = fusion | (color_likelihood01 > 0.6)
        fusion = (fusion.astype(float) + (color_likelihood01 > 0.6).astype(float)) >= 1

        return fusion, None, None

    return cw_mtf_fusion(preprocessed_image, manual_threshold, color_likelihood01)


# ---------------- MORPHOLOGY ----------------
def refine_mask(binary_mask):
    cleaned = morphology.opening(binary_mask, morphology.disk(2))
    cleaned = morphology.closing(cleaned, morphology.disk(3))
    cleaned = morphology.remove_small_objects(cleaned, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=500)
    cleaned = morphology.dilation(cleaned, morphology.disk(1))
    return cleaned


# ---------------- WATERSHED ----------------
def watershed_split(binary_mask):
    dist = ndi.distance_transform_edt(binary_mask)
    coords = feature.peak_local_max(dist, footprint=np.ones((25, 25)), labels=binary_mask)

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        labels = morphology.label(binary_mask)
        return labels, binary_mask

    labels = segmentation.watershed(-dist, markers, mask=binary_mask)
    return labels, labels > 0


# ---------------- METRICS ----------------
def colony_metrics_from_labels(labels):
    props = measure.regionprops(labels)

    areas = [p.area for p in props if p.area > 20]
    if not areas:
        return [], {}

    circularity, equiv_diam = [], []

    for p in props:
        if p.area <= 20:
            continue

        perim = max(p.perimeter, 1e-8)
        circularity.append(4 * np.pi * p.area / (perim ** 2))
        equiv_diam.append(p.equivalent_diameter)

    return areas, {
        "count": len(areas),
        "total_area_px": float(np.sum(areas)),
        "mean_area_px": float(np.mean(areas)),
        "median_area_px": float(np.median(areas)),
        "mean_circularity": float(np.mean(circularity)),
        "mean_equiv_diameter_px": float(np.mean(equiv_diam)),
    }


# ---------------- SAVE IMAGE ----------------
def save_image(path, image, cmap=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap if image.ndim == 2 else None)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------------- MAIN FUNCTION ----------------
def analyze_image(
    image_path,
    method="CW-MTF (Novel)",
    resize_factor=1.0,
    manual_threshold=0.5,
    use_watershed_split=True
):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    image = io.imread(image_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)

    # ---------------- RESIZE ----------------
    resized = transform.resize(
        image,
        (int(image.shape[0]*resize_factor), int(image.shape[1]*resize_factor)),
        mode="reflect",
        anti_aliasing=True
    )

    # ---------------- PREPROCESS ----------------
    hsv = color.rgb2hsv(resized)
    gray = color.rgb2gray(resized)
    preprocessed = gaussian(gray, sigma=1)

    # ---------------- COLOR ----------------
    color_likelihood = get_color_likelihood_mask(hsv)

    # ---------------- SEGMENTATION ----------------
    binary, uncertainty, weights = get_binary_mask(
        preprocessed,
        color_likelihood,
        manual_threshold,
        method
    )

    # ---------------- MORPHOLOGY ----------------
    cleaned = refine_mask(binary)

    # ---------------- WATERSHED ----------------
    if use_watershed_split:
        labels, separated = watershed_split(cleaned)
    else:
        labels = morphology.label(cleaned)
        separated = cleaned

    # ---------------- FINAL SEGMENTED ----------------
    segmented = resized.copy()
    segmented[~separated] = 0

    # ---------------- METRICS ----------------
    _, metrics = colony_metrics_from_labels(labels)

    # ---------------- SAVE ----------------
    result_id = str(uuid.uuid4())
    os.makedirs("results", exist_ok=True)

    def p(name): return f"results/{result_id}_{name}.png"

    save_image(p("resized"), resized)
    save_image(p("gray"), gray, "gray")
    save_image(p("preprocessed"), preprocessed, "gray")
    save_image(p("color"), color_likelihood, "gray")
    save_image(p("binary"), binary.astype(np.uint8)*255, "gray")
    save_image(p("refined"), separated.astype(np.uint8)*255, "gray")
    save_image(p("segmented"), segmented)

    # ✅ CW-MTF EXTRA OUTPUT
    if method == "CW-MTF (Novel)" and uncertainty is not None:
        save_image(p("uncertainty"), uncertainty, "gray")

    # ---------------- WEIGHTS ----------------
    if weights:
        wo, wa, wm, wc = weights
        weights_dict = {
            "otsu": wo,
            "adaptive": wa,
            "manual": wm,
            "color": wc
        }
    else:
        weights_dict = None

    # ---------------- IMAGE OUTPUT ----------------
    images = {
        "resized": p("resized"),
        "gray": p("gray"),
        "preprocessed": p("preprocessed"),
        "color": p("color"),
        "binary": p("binary"),
        "refined": p("refined"),
        "segmented": p("segmented"),
    }

    # ✅ Add uncertainty ONLY for CW-MTF
    if method == "CW-MTF (Novel)" and uncertainty is not None:
        save_image(p("uncertainty"), uncertainty, "gray")
        images["uncertainty"] = p("uncertainty")

    return {
        "id": result_id,
        "method": method,
        "metrics": metrics,
        "weights": weights_dict,
        "images": images
    }
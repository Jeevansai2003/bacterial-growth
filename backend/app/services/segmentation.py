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
    attempts = 3
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
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
        soft = gaussian(hard.astype(np.float32), sigma=1.0)
        return soft

    centers, _, _ = res
    hue_dist = np.abs(centers[:, 0] - target_hue)
    penalty = (centers[:, 1] < 0.25).astype(np.float32) * 0.25 + (centers[:, 2] < 0.25).astype(np.float32) * 0.25
    score = hue_dist + penalty
    chosen = int(np.argmin(score))
    c = centers[chosen]

    H, S, V = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]
    dh = np.minimum(np.abs(H - c[0]), 1.0 - np.abs(H - c[0]))
    ds = S - c[1]
    dv = V - c[2]

    sh, ss, sv = 0.06, 0.25, 0.25
    d2 = (dh / sh) ** 2 + (ds / ss) ** 2 + (dv / sv) ** 2
    likelihood = np.exp(-0.5 * d2).astype(np.float32)

    likelihood = gaussian(likelihood, sigma=1.0)
    likelihood = np.clip(likelihood, 0.0, 1.0)
    return likelihood


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


def cw_mtf_fusion(preprocessed01, manual_t, color_likelihood01):
    t_otsu = filters.threshold_otsu(preprocessed01)
    otsu_mask = preprocessed01 > t_otsu

    adaptive_th = filters.threshold_local(preprocessed01, block_size=15)
    adaptive_mask = preprocessed01 > adaptive_th

    manual_mask = preprocessed01 > manual_t

    wo = otsu_confidence(preprocessed01)
    wa = adaptive_confidence(preprocessed01, block=15)
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


def get_binary_mask(preprocessed_image, color_likelihood01, manual_threshold, method):
    if method == "Otsu":
        binary_mask = preprocessed_image > filters.threshold_otsu(preprocessed_image)
        return binary_mask, None, None

    if method == "Adaptive":
        binary_mask = preprocessed_image > filters.threshold_local(preprocessed_image, block_size=15)
        return binary_mask, None, None

    if method == "Manual":
        binary_mask = preprocessed_image > manual_threshold
        return binary_mask, None, None

    if method == "Majority Fusion":
        otsu_mask = preprocessed_image > filters.threshold_otsu(preprocessed_image)
        adaptive_mask = preprocessed_image > filters.threshold_local(preprocessed_image, block_size=15)
        manual_mask = preprocessed_image > manual_threshold

        fusion = (
            otsu_mask.astype(np.uint8) +
            adaptive_mask.astype(np.uint8) +
            manual_mask.astype(np.uint8)
        ) >= 2

        fusion = fusion | (color_likelihood01 > 0.6)
        return fusion, None, None

    fusion_mask, uncertainty_map, weights = cw_mtf_fusion(
        preprocessed_image,
        manual_threshold,
        color_likelihood01
    )
    return fusion_mask, uncertainty_map, weights


def refine_mask(binary_mask):
    cleaned = morphology.opening(binary_mask, morphology.disk(2))
    cleaned = morphology.closing(cleaned, morphology.disk(3))
    cleaned = morphology.remove_small_objects(cleaned, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=500)
    cleaned = morphology.dilation(cleaned, morphology.disk(1))
    return cleaned


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
    separated = labels > 0
    return labels, separated


def colony_metrics_from_labels(labels):
    props = measure.regionprops(labels)
    areas = [p.area for p in props if p.area > 20]
    if not areas:
        return [], {}

    circularity = []
    equiv_diam = []

    for p in props:
        if p.area <= 20:
            continue
        perim = p.perimeter if p.perimeter > 1e-8 else 1e-8
        circ = 4.0 * np.pi * p.area / (perim ** 2)
        circularity.append(circ)
        equiv_diam.append(p.equivalent_diameter)

    metrics = {
        "count": len(areas),
        "total_area_px": float(np.sum(areas)),
        "mean_area_px": float(np.mean(areas)),
        "median_area_px": float(np.median(areas)),
        "mean_circularity": float(np.mean(circularity)) if circularity else 0.0,
        "mean_equiv_diameter_px": float(np.mean(equiv_diam)) if equiv_diam else 0.0,
    }
    return areas, metrics


def save_image(path, image, cmap=None):
    plt.figure(figsize=(6, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap=cmap if cmap else "gray")
    else:
        plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def analyze_image(
    image_path,
    resize_factor=1.0,
    manual_threshold=0.5,
    use_watershed_split=True,
    method="CW-MTF (Novel)",
):
    if method not in SUPPORTED_METHODS:
        method = "CW-MTF (Novel)"

    image = io.imread(image_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)

    resized_image = transform.resize(
        image,
        (
            int(image.shape[0] * resize_factor),
            int(image.shape[1] * resize_factor)
        ),
        mode="reflect",
        anti_aliasing=True
    )

    hsv_image = color.rgb2hsv(resized_image)
    color_likelihood01 = get_color_likelihood_mask(hsv_image)

    gray_image = color.rgb2gray(resized_image)
    preprocessed_image = gaussian(gray_image, sigma=1)

    binary_image, uncertainty_map, weights = get_binary_mask(
        preprocessed_image=preprocessed_image,
        color_likelihood01=color_likelihood01,
        manual_threshold=manual_threshold,
        method=method,
    )

    cleaned_image = refine_mask(binary_image)

    if use_watershed_split:
        labels, separated_mask = watershed_split(cleaned_image)
    else:
        labels = morphology.label(cleaned_image)
        separated_mask = cleaned_image

    segmented_image = resized_image.copy()
    segmented_image[~separated_mask] = 0

    _, metrics = colony_metrics_from_labels(labels)

    result_id = str(uuid.uuid4())
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    resized_path = os.path.join(results_dir, f"{result_id}_resized.png")
    mask_path = os.path.join(results_dir, f"{result_id}_mask.png")
    uncertainty_path = os.path.join(results_dir, f"{result_id}_uncertainty.png")
    segmented_path = os.path.join(results_dir, f"{result_id}_segmented.png")

    save_image(resized_path, resized_image)
    save_image(mask_path, separated_mask.astype(np.uint8) * 255, cmap="gray")

    if uncertainty_map is not None:
        uncertainty_vis = uncertainty_map * separated_mask.astype(np.float32)
        save_image(uncertainty_path, uncertainty_vis, cmap="gray")
        uncertainty_result_path = uncertainty_path.replace("\\", "/")
    else:
        uncertainty_result_path = None

    save_image(segmented_path, segmented_image)

    weights_response = None
    if weights is not None:
        wo, wa, wm, wc = weights
        weights_response = {
            "otsu": float(wo),
            "adaptive": float(wa),
            "manual": float(wm),
            "color": float(wc)
        }

    return {
        "result_id": result_id,
        "method": method,
        "metrics": metrics,
        "weights": weights_response,
        "images": {
            "resized": resized_path.replace("\\", "/"),
            "mask": mask_path.replace("\\", "/"),
            "uncertainty": uncertainty_result_path,
            "segmented": segmented_path.replace("\\", "/")
        }
    }
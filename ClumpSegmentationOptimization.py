###################################################################################################
###### This script optimizes the parameters of a watershed / morpholigical image segmentation ######
###################################################################################################


import numpy as np
import rasterio
import geopandas as gpd
import optuna
from pathlib import Path
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import filters, morphology
from skimage.filters import sobel
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import disk, dilation, h_minima
from skimage.measure import regionprops
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# Set Directories
base_dir = Path("data/input")
raster_dir = Path("data/input")
ground_truth_dir = Path("data/input")

# Define batch files
batch_files = [
  (raster_dir / f"selectclumps_batch{i}.tif", ground_truth_dir / f"centroid_batch{i}.shp")
  for i in range(1, 31) # Adjust range as needed. This is in accordance with the batches you created with the R script. Centroids were created in QGIS.
]

# Load raster data. Downscale as needed.
def load_raster(raster_path, downscale_factor=0):
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        new_h = src.height // downscale_factor if downscale_factor else src.height
        new_w = src.width // downscale_factor if downscale_factor else src.width
        transform = src.transform * src.transform.scale(src.width / new_w, src.height / new_h)
        data = src.read(1, out_shape=(new_h, new_w), resampling=rasterio.enums.Resampling.average)
    return data, transform, nodata

# Load and prepare ground truth data
def load_ground_truth(shapefile, reference_raster):
    if not shapefile.exists() or not reference_raster.exists():
        return None
    with rasterio.open(reference_raster) as src:
        crs = src.crs
    gdf = gpd.read_file(shapefile).to_crs(crs)
    return np.array([(pt.x, pt.y) for pt in gdf.geometry if pt.geom_type == "Point"])

# Apply segmentation
def apply_segmentation(raster_path, gamma, sigma, threshold_type, erosion_disk, dilation_disk, min_dist_frac, h_minima_frac):
    elevation, transform, nodata = load_raster(raster_path)
    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)
    elevation = np.nan_to_num(elevation, nan=np.nanmin(elevation))
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    elevation = np.power(elevation, gamma)
    elevation = ndi.gaussian_filter(elevation, sigma=sigma)
    gradient = sobel(elevation)
    thresh = filters.threshold_otsu(elevation) if threshold_type == "otsu" else filters.threshold_li(elevation)
    binary = elevation > thresh
    binary = morphology.erosion(binary, disk(erosion_disk))
    distance = ndi.distance_transform_edt(binary * gradient)
    min_dist = min_dist_frac * np.max(distance)
    peaks = peak_local_max(distance, footprint=np.ones((4, 4)), labels=binary)
    filtered = peaks[distance[tuple(peaks.T)] > min_dist]
    markers = np.zeros_like(distance, dtype=bool)
    markers[tuple(filtered.T)] = True
    markers = dilation(markers, disk(dilation_disk))
    markers, _ = ndi.label(markers)
    labels = watershed(-distance, markers, mask=binary)
    h_min = h_minima(-distance, h_minima_frac * np.max(distance))
    hwt_markers, _ = ndi.label(h_min)
    labels_hwt = watershed(-distance, hwt_markers, mask=binary)
    centroids = np.array([region.centroid for region in regionprops(labels_hwt)])
    return np.array([rasterio.transform.xy(transform, y, x) for y, x in centroids])

# Use the Hungarian algorithm to match detected Centroids. Adjust buffer as needed.
def compute_global_iou(gt_centroids_list, detected_centroids_list, transform, buffer_radius=49): # Buffer value currently in pixels.
    all_gt_world = np.vstack(gt_centroids_list) if gt_centroids_list else np.empty((0, 2))
    all_det_world = np.vstack(detected_centroids_list) if detected_centroids_list else np.empty((0, 2))
    gt_count, det_count = len(all_gt_world), len(all_det_world)
    if gt_count == 0 or det_count == 0:
        return 0.0, gt_count, det_count, 0.0

    transform_inv = ~transform
    all_gt_px = np.array([transform_inv * pt for pt in all_gt_world])
    all_det_px = np.array([transform_inv * pt for pt in all_det_world])
    dist = pairwise_distances(all_gt_px, all_det_px)
    row_ind, col_ind = linear_sum_assignment(dist)
    matched = sum(dist[r, c] < buffer_radius for r, c in zip(row_ind, col_ind))

    union = gt_count + det_count - matched
    iou = matched / union if union > 0 else 0.0
    if det_count > gt_count:
        iou *= (gt_count / det_count)
    count_acc = max(0, 1 - abs(gt_count - det_count) / max(gt_count, 1))
    final_score = 0.5 * count_acc + 0.5 * iou # Adjust final score calculation as needed.
    return iou, gt_count, det_count, final_score

# Optimization with Optuna objective. Adjust hyperparameter ranges as needed.
def objective(trial, enable_visuals=False):
    gamma = trial.suggest_float("gamma", 0.5, 1.5)
    sigma = trial.suggest_float("sigma", 0.1, 1.5)
    threshold_type = trial.suggest_categorical("threshold_type", ["otsu"])
    erosion_disk = trial.suggest_int("erosion_disk", 0, 3)
    dilation_disk = trial.suggest_int("dilation_disk", 0, 4)
    min_dist_frac = trial.suggest_float("min_distance_fraction", 0.1, 0.5)
    h_minima_frac = trial.suggest_float("h_minima_fraction", 0.005, 0.4)
    buffer_radius = 49

    batch_scores = []
    matched_gt_total, gt_total, det_total = 0, 0, 0

    for idx, (raster_path, gt_path) in tqdm(enumerate(batch_files), total=len(batch_files), desc="Processing batches"):
        if not raster_path.exists() or not gt_path.exists():
            continue
        gt = load_ground_truth(gt_path, raster_path)
        if gt is None:
            continue
        elevation_map, transform_img, _ = load_raster(raster_path)
        det = apply_segmentation(raster_path, gamma, sigma, threshold_type, erosion_disk, dilation_disk, min_dist_frac, h_minima_frac)
        if det is None or len(det) == 0:
            print(f"Batch {idx + 1}: no detections.")
            continue
        iou, gt_count, det_count, score = compute_global_iou([gt], [det], transform_img, buffer_radius)
        batch_scores.append(score)
        gt_total += gt_count
        det_total += det_count

        transform_inv = ~transform_img
        gt_px = np.array([transform_inv * pt for pt in gt])
        det_px = np.array([transform_inv * pt for pt in det])
        dist_matrix = pairwise_distances(gt_px, det_px)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_pairs = [(r, c) for r, c in zip(row_ind, col_ind) if dist_matrix[r, c] < buffer_radius]
        matched_gt_total += len({r for r, _ in matched_pairs})

    final_score = np.mean(batch_scores) if batch_scores else 0.0

    if gt_total > 0:
        global_iou = matched_gt_total / (gt_total + det_total - matched_gt_total)
        global_count_acc = max(0, 1 - abs(gt_total - det_total) / max(gt_total, 1))
        global_final_score = 0.5 * global_count_acc + 0.5 * global_iou
        print(f"\n Global Final Score: {global_final_score:.3f}")
        print(f" Global IoU: {global_iou:.3f}")
        print(f" Global Count Accuracy: {global_count_acc:.3f}")
        print(f" Total Ground Truth: {gt_total}")
        print(f" Total Detections: {det_total}")
    else:
        global_final_score = 0.0
        print("\n No ground truth points found across batches.")

    return global_final_score

# Run optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="per_batch_opt")
    study.optimize(lambda trial: objective(trial, enable_visuals=True), n_trials=500) # Adjust number of trials as needed.
    print(" Best params:", study.best_params)

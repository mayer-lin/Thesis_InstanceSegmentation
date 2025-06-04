#########################################################################################################
###### Using optimized hyperparameters, this script applies watershed / morpholigical segmentation ######
#########################################################################################################
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
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
import matplotlib.pyplot as plt

# Set Directories
raster_dir = Path("data/input")
ground_truth_dir = Path("data/input")
results_dir = Path("data/input")
fig_dir = Path("data/results")
results_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# Define batch files. Adjust as needed. GT centroids were calculated in QGIS.
batch_files = [
    (raster_dir / f"selectclumps_batch{i}.tif", ground_truth_dir / f"centroid_batch{i}.shp")
    for i in range(1, 3)
]

# Prepare raster data. Downscale as needed.
def load_raster(raster_path, downscale_factor=0):
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        new_h = src.height // downscale_factor if downscale_factor else src.height
        new_w = src.width // downscale_factor if downscale_factor else src.width
        transform = src.transform * src.transform.scale(src.width / new_w, src.height / new_h)
        data = src.read(
            1,
            out_shape=(new_h, new_w),
            resampling=rasterio.enums.Resampling.average
        )
    return data, transform, nodata

# Prepare ground truth centroids.
def load_ground_truth(shapefile, reference_raster):
    if not shapefile.exists() or not reference_raster.exists():
        return None
    with rasterio.open(reference_raster) as src:
        crs = src.crs
    gdf = gpd.read_file(shapefile).to_crs(crs)
    return np.array([(pt.x, pt.y) for pt in gdf.geometry if pt.geom_type == "Point"])

# Apply segmentation
def apply_segmentation(raster_path,gamma, sigma,threshold_type,erosion_disk,dilation_disk,min_dist_frac,h_minima_frac):
    elevation, transform, nodata = load_raster(raster_path)
    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)
    elevation = np.nan_to_num(elevation, nan=np.nanmin(elevation))
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    elevation = np.power(elevation, gamma)
    elevation = ndi.gaussian_filter(elevation, sigma=sigma)
    gradient = sobel(elevation)
    thresh = (
        filters.threshold_otsu(elevation)
        if threshold_type == "otsu"
        else filters.threshold_li(elevation)
    )
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
    _ = watershed(-distance, markers, mask=binary)
    h_min = h_minima(-distance, h_minima_frac * np.max(distance))
    hwt_markers, _ = ndi.label(h_min)
    labels_hwt = watershed(-distance, hwt_markers, mask=binary)
    centroids = np.array([region.centroid for region in regionprops(labels_hwt)])
    return np.array([rasterio.transform.xy(transform, y, x) for y, x in centroids])

# Use the Hungarian algorithm to match detected Centroids. Adjust buffer as needed.
def compute_global_iou(gt_centroids_list, detected_centroids_list, transform, buffer_radius=49):
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
    final_score = 0.5 * count_acc + 0.5 * iou
    return iou, gt_count, det_count, final_score

# Run segmentation & evaluation with fixed parameters
def run_with_params(params, enable_visuals=True):
    gamma = params["gamma"]
    sigma = params["sigma"]
    threshold_type = params["threshold_type"]
    erosion_disk = params["erosion_disk"]
    dilation_disk = params["dilation_disk"]
    min_dist_frac = params["min_distance_fraction"]
    h_minima_frac = params["h_minima_fraction"]
    buffer_radius = 49

    batch_scores = []
    batch_metrics = []

    for idx, (raster_path, gt_path) in enumerate(batch_files):
        batch_id = idx + 1
        print(f" Processing batch {batch_id}:")
        print(f"    Raster exists: {raster_path.exists()} | GT exists: {gt_path.exists()}")
        if not raster_path.exists() or not gt_path.exists():
            continue

        gt = load_ground_truth(gt_path, raster_path)
        if gt is None:
            print(f"No valid ground truth for batch {batch_id}.")
            continue
        else:
            print(f"    # GT loaded: {len(gt)} points")

        elevation_map, transform_img, _ = load_raster(raster_path)
        det = apply_segmentation(
            raster_path,
            gamma,
            sigma,
            threshold_type,
            erosion_disk,
            dilation_disk,
            min_dist_frac,
            h_minima_frac
        )
        if det is None or len(det) == 0:
            print(f" Batch {batch_id}: no detections.")
            continue

        iou, gt_count, det_count, score = compute_global_iou(
            [gt], [det], transform_img, buffer_radius
        )
        count_acc = max(0, 1 - abs(gt_count - det_count) / max(gt_count, 1))
        print(f"      Batch {batch_id} — IoU: {iou:.3f}, Count Acc: {count_acc:.3f}, Score: {score:.3f}")

        transform_inv = ~transform_img
        gt_px = np.array([transform_inv * pt for pt in gt])
        det_px = np.array([transform_inv * pt for pt in det])
        dist_matrix = pairwise_distances(gt_px, det_px)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_pairs = [
            (r, c)
            for r, c in zip(row_ind, col_ind)
            if dist_matrix[r, c] < buffer_radius
        ]
        matched = len(matched_pairs)
        matched_gt = set(r for r, _ in matched_pairs)
        matched_det = set(c for _, c in matched_pairs)
        missed_gt = len(gt) - len(matched_gt)
        extra_det = len(det) - len(matched_det)
        print(f"      Matched: {matched}, Missed GT: {missed_gt}, Extra Det: {extra_det}")

        batch_scores.append(score)
        batch_metrics.append({
            "batch": batch_id,
            "iou": round(iou, 3),
            "count_accuracy": round(count_acc, 3),
            "gt_count": gt_count,
            "detected_count": det_count,
            "matched": matched,
            "missed_gt": missed_gt,
            "extra_detected": extra_det,
            "score": round(score, 3)
        })

    final_score = np.mean(batch_scores) if batch_scores else 0.0
    print(f"\n Overall Final Score: {final_score:.3f}\n")

    if enable_visuals and batch_metrics:
        df = pd.DataFrame(batch_metrics)
        df["site"] = df["batch"].apply(lambda x: "Stet 1" if x <= 20 else "Jona 1") # Adjust this according to your site batch numbers
        df.to_csv(results_dir / "segmentation_summary.csv", index=True) # rename as needed
        print(f"Saved summary to {results_dir / 'segmentation_summary.csv'}") # rename as needed

        # Observed vs Predicted Scatter
        plt.figure(figsize=(7, 6))
        stet_df = df[df["site"] == "Stet 1"]
        plt.scatter(
            stet_df["gt_count"],
            stet_df["detected_count"],
            label="Stet 1",
            alpha=0.9,
            edgecolors='k',
            color="#fde725"
        )
        jona_df = df[df["site"] == "Jona 1"]
        plt.scatter(
            jona_df["gt_count"],
            jona_df["detected_count"],
            label="Jona 1",
            alpha=0.9,
            edgecolors='k',
            color="#414487"
        )
        max_val = max(df["gt_count"].max(), df["detected_count"].max()) + 10
        plt.plot([0, max_val], [0, max_val], 'grey', linestyle='--', label='1:1 Line')
        plt.xlabel("Ground Truth Count (Observed)")
        plt.ylabel("Detected Count (Predicted)")
        plt.title("Observed vs. Predicted Counts by Site")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_dir / "observed_vs_predicted_scatter.png") #Rename as needed
        plt.show()
        plt.close()

    return final_score

# SET HYPERPARAMETERS HERE!! You can use the best performing hyperparameters from the optimization trial.
if __name__ == "__main__":
    fixed_params = {
        "gamma": 0.5168518310522704,
        "sigma": 0.729704864886184,
        "threshold_type": "otsu",
        "erosion_disk": 3,
        "dilation_disk": 2,
        "min_distance_fraction": 0.174633335416156,
        "h_minima_fraction": 0.0914933448375912
    }

    print(" Running segmentation on all batches with set hyperparameters...")
    run_with_params(fixed_params, enable_visuals=True)

    summary_path = results_dir / "segmentation_summary.csv" # Rename as needed
    if summary_path.exists():
        print("\n Per-batch Results Summary:")
        summary_df = pd.read_csv(summary_path)
        print(summary_df.to_string(index=False))
    else:
        print("No per-batch summary CSV found.")
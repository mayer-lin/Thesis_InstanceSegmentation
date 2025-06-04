Master's Thesis title: "A Machine Learning and Remote Sensing Framework for Instance-Level Shrub Segmentation in a Biodiversity Hotspot"

Author: Mayerlin Estrella Fischbach Barria

Description: A compilation of the essential scripts created to accomplish the work in my master's thesis. 

Python 3.12.6
R version 4.3.3 (2024-02-29 ucrt) -- "Angel Food Cake"


===============================================================================
RasterHeightCalculations (R Script)
===================================

This script computes normalized elevation values for segmented polygons (e.g., plant clumps) 
by using a buffered reference generated from a Digital Elevation Model (DEM). It assigns a new zero 
elevation value per polygon based on the lowest ground point within the created buffer.

INPUT REQUIREMENTS:

- DEM raster file (e.g., jona1_DEM_raster.tif)
- Polygon shapefile (e.g., jona1_clumps_polygons.shp)

All spatial files should use the same coordinate reference system, but the script 
should reproject if needed.

PROCESS OVERVIEW:

1. Polygons are divided into assigned batches to optimize memory usage.
2. For each polygon:
   - A buffer ring is generated to define its surrounding ground area.
   - The 1% elevation quantile from this buffer is used as a reference.
   - Elevation values inside the polygon are normalized relative to this new reference zero.
3. Outputs are stored as one raster per batch with the new relative height values.


OUTPUT FILES:

- Relative elevation rasters per batch (e.g., jona1_Batch1_elevation.tif)
- Optional shapefiles of batch polygons for record-keeping.

DEPENDENCIES:

- R package: terra

AUTHORSHIP:

Mayerlin Fischbach Barria
Adapted from a script by: Dr. Frank Schurr


===============================================================================
RandomForestPolygonClassifier (Python Script)
=============================================

This script utilizes a calibrated Random Forest classifier to classify polygons as either an individual or a cluster based on geometric attributes calculated in QGIS. It loads an Excel file containing the polygon metrics, cleans the data, trains and evaluates the model according to set thresholds, and produces a classification report. There is also the possibility to apply the classifier to unlabeled polygons. 

INPUT REQUIREMENTS:

- Excel file (e.g., combined_geoms.xlsx) exported from QGIS containing:
  - A column “Indv_or_cl” indicating if a polygon is of an individual or a cluster.
  - Numeric float columns for area, perimeter, roundness, perim2area, 
    convexhull_ratio, and dist2nearest_indiv. These values were calculated in QGIS
    using the "calculate Geometry" plugin. 

PROCESS OVERVIEW:

1. Load and clean data.
2. Split the data into training and testing sets with an assigned/fixed random state.
3. Train the model with set calibration. 
4. Set and apply a thresholding logic to create predictions.
5. Evaluate the outcomes by printing a classification report on the testing data.
6. Create predictions for any unlabeled polygons. 

OUTPUT FILES / RESULTS:

- Classification report on testing data printed in the console.
- Option to save the predictions for the entire dataset as an Excel file 
  (e.g., classified_polygons_random_forest.xlsx) 

DEPENDENCIES:

- pandas  
- scikit-learn (sklearn)

AUTHORSHIP:
Mayerlin Fischbach Barria


===============================================================================
ClumpSegmentationOptimization (Python Script)
=============================================

This script segments elevation rasters of plant clumps via morphological and watershed operations. It optimizes segmentation hyperparameters with Optuna by comparing detected centroids to ground‐truth points and following a set scoring logic.

INPUT REQUIREMENTS:

- Raster files (e.g., selectclumps_batch1.tif)
- Ground‐truth centroid shapefiles (created in QGIS) (e.g., centroid_batch1.shp)

PROCESS OVERVIEW:

1. Load Data.
2. Segmentation:  
   - Normalize and smooth elevation.  
   - Generate binary mask, apply erosion and distance transform.  
   - Find local maxima, create markers, and perform two watershed passes (standard + 
     h‐minima).  
3. Extract detected centroids.  
4. Match GT and detected centroids using the Hungarian algorithm and a set buffer.
5. Compute IoU and count accuracy, and combine into a final score per batch and 
   globally.  
6. Optimization!! Optuna optimizes hyperparameters based on a designated range of 
   values for a set number of trials. For each trial, run steps 1–5 across all
   batches, and return the global final score. 
7. After all trials have been completed, report the top-scoring parameters.

OUTPUT FILES / RESULTS:

- Console shows per‐batch skips/detections and final global IoU/count accuracy.  
- Best hyperparameters printed at the end.  

DEPENDENCIES:

- numpy, rasterio, geopandas  
- scipy (ndimage, optimize), scikit-image  
- scikit-learn, tqdm, optuna

AUTHORSHIP:

Mayerlin Fischbach Barria

===============================================================================
Segmentation (Python Script)
======================================

This script segments elevation rasters of plant clumps via morphological and watershed operations, then evaluates detections against ground‐truth centroids. It saves per‐batch metrics in a CSV and plots observed vs. predicted counts. If GT data is missing, the script will need to be adjusted to leave out evaluation steps. 

INPUT REQUIREMENTS:

- Raster files (e.g., selectclumps_batch1.tif)
- Ground‐truth centroid shapefiles (created in QGIS) (e.g., centroid_batch1.shp)

PROCESS OVERVIEW:

1. Load Data. Set hyperparameters (starting at line 238).
2. Segmentation:  
   - Normalize and smooth elevation.  
   - Generate binary mask, apply erosion and distance transform.  
   - Find local maxima, create markers, and perform two watershed passes (standard + 
     h‐minima).
   - Extract detected centroids.  
3. Match GT and detected centroids using the Hungarian algorithm and a set buffer.
4. Compute IoU and count accuracy.
6. Save per‐batch summary CSV to results directory. 
7. Plot and save observed vs. predicted counts scatter to figure directory. 


OUTPUT FILES:

- Console logs: file checks, per‐batch IoU/count accuracy, match details, overall score.  
- CSV file with batch, iou, count_accuracy, GT_count, detected_count, matched, missed_gt, extra_detected, and score information.
- PNG of observed vs predicted scatter plot by site.

DEPENDENCIES:

- numpy, rasterio, geopandas, pandas  
- scipy (ndimage, optimize), scikit-image, scikit-learn, matplotlib

AUTHORSHIP:

Mayerlin Fischbach Barria


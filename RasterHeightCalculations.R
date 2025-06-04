#####################################################################################
# Title: Raster Height Calculations                                                 #
# Description: This script computes relative elevation values for polygons across   #
#              batches using a buffered DEM reference. Essentially it creates a     #
#              new zero depending on the lowest point within the buffered range     #
#              of a given polygon.                                                  #
#                                                                                   #
# Inputs:                                                                           #
#   - DEM raster file (e.g. jona1_DEM_raster.tif)                                        #
#   - Polygon shapefile (e.g. jona1_clumps_polygons.shp)                            #
#                                                                                   #
# Outputs:                                                                          #
#   - Normalized per-batch elevation rasters (.tif)                                            #
#   - Optional batch shapefiles (.shp)                                              #
#                                                                                   #
# Author: Mayerlin Fischbach Barria, adapted from Dr. Frank Schurr                  #
# Dependencies: terra                                                               #
#####################################################################################

# Load library
library(terra)

# Directories 
input_dir <- "data/input"
output_dir <- "results/output"

# Site Name
site <- "jona1" # Adjust as needed

# Set file paths
dem_path <- file.path(input_dir, "jona1_DEM_raster.tif") # Replace as needed 
polygon_path <- file.path(input_dir, "jona1_clumps_polygons.shp") # Replace with identified clumps of interest

# Load input data
dem <- rast(dem_path)
plants <- vect(polygon_path)

# Ensure CRS alignment of vector and raster data. Optional but was at some point helpful for debugging. 
if (!crs(dem) == crs(plants)) {
  plants <- project(plants, crs(dem))
}

# Create batch size
num_batches <- 10
batch_size <- ceiling(nrow(plants) / num_batches)

# Specify which batch to start from
start_batch <- 1

# Initialize temp_files. This is called to eventually clear temporary files and reduce memory demands. 
temp_files <- c()

# Process each batch *starting from start_batch*
for (b in start_batch:num_batches) {  
  
  print(paste("Processing Batch", b, "of", num_batches))
  
  batch_indices <- ((b - 1) * batch_size + 1):min(b * batch_size, nrow(plants))

  batch_polygons <- plants[batch_indices, ]
  
  # Save batch polygons to a separate shapefile. Optional, use only if you want the polygons saved according to their assigned batch. 
  batch_poly_path <- paste0(output_dir, "/", site, "_Batch", b, "_polygons.shp")
  writeVector(batch_polygons, batch_poly_path, overwrite = TRUE)
  print(paste("Saved batch", b, "polygons at:", batch_poly_path))
  
  # Creation of the new rasters
  batch_raster <- rast(extent = ext(dem), resolution = res(dem), crs = crs(dem), vals = NA)
  
  for (i in batch_indices) {
    
    print(paste("Processing polygon", i, "of", nrow(plants)))
    
    poly <- plants[i, ]
    
    buff.outer <- buffer(poly, width = 1) # Adjust as needed
    buff.inner <- buffer(poly, width = 0.5) # Adjust as needed 
    
    # Subtract inner buffer from outer buffer to get ring 
    buff <- erase(buff.outer, buff.inner)
    
    # Crop DEM to the extent of the outer buffer
    dem.crop <- crop(dem, buff.outer)
    
    # Mask the cropped DEM with the ring buffer
    dem.ref <- mask(dem.crop, buff)
    
    # Calculate 1% quantile for reference ground height
    z.ref <- quantile(values(dem.ref), probs = 0.01, na.rm = TRUE)
    
    # Mask DEM to extract elevation values within the polygon
    dem.mask <- mask(dem.crop, poly)
    
    # Adjust elevation values relative to the reference ground height
    dem.mask <- dem.mask - z.ref
    
    # Align the masked raster to the output raster dimensions
    dem.mask <- resample(dem.mask, batch_raster, method = "bilinear")
    
    # Accumulate changes in the batch raster
    batch_raster <- cover(batch_raster, dem.mask)
    
    # Save temporary raster for later cleanup
    temp_file <- paste0(output_dir, "/temp_dem_", b, "_", i, ".tif")
    writeRaster(dem.mask, temp_file, overwrite = TRUE)
    temp_files <- c(temp_files, temp_file)  
  }
  
  # Save the batch raster to a separate file
  batch_output_path <- paste0(output_dir, "/", site, "_Batch", b, "_elevation.tif")
  writeRaster(batch_raster, batch_output_path, overwrite = TRUE)
  print(paste("Batch", b, "raster saved at:", batch_output_path))
  
  # Delete temporary files
  for (temp_file in temp_files) {
    if (file.exists(temp_file)) {
      file.remove(temp_file)
      print(paste("Deleted temporary file:", temp_file))
    }
  }
  
  # Release memory by clearing variables
  rm(batch_raster, dem.mask, dem.ref, dem.crop, buff, buff.outer, buff.inner, poly, batch_polygons)
  gc()
}

print("All raster layers and shapefiles for this batch have been saved.")


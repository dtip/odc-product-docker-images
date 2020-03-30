################
# NDVI Anomaly #
################

import dask
import numpy as np

from datacube_utilities.dc_mosaic import create_max_ndvi_mosaic, create_median_mosaic
from datacube_utilities.createindices import NDVI

from masking import mask_good_quality


def _query(
    product,
    query_x_from,
    query_x_to,
    query_y_from,
    query_y_to,
    time_from,
    time_to,
    output_crs,
    query_crs,
    dask_chunk_size,
):

    time_extents = (time_from, time_to)

    data_bands = ["red", "green", "blue", "nir", "swir1", "swir2"]
    mask_bands = [
        "pixel_qa" if product.startswith("ls") else "coastal_aerosol",
        "scene_classification",
    ]

    if product.startswith("ls"):
        resolution = (-30, 30)
        group_by = "solar_day"
        water_product = product[:3] + "_water_classification"
    else:
        resolution = (-10, 10)
        group_by = "time"
        # TODO: Change when S2 WOFS ready
        water_product = "SENTINEL_2_PRODUCT DEFS"

    query = {}

    query["product"] = product
    query["time"] = time_extents
    query["output_crs"] = output_crs
    query["resolution"] = resolution
    query["measurements"] = data_bands + mask_bands
    query["group_by"] = group_by
    query["dask_chunks"] = {"x": int(dask_chunk_size), "y": int(dask_chunk_size)}

    if query_crs != "EPSG:4326":
        query["crs"] = query_crs

    query["x"] = (float(query_x_from), float(query_x_to))
    query["y"] = (float(query_y_from), float(query_y_to))

    return query, water_product


def process_ndvi_anomaly(
    dc,
    baseline_product,
    analysis_product,
    mosaic_type,
    query_x_from,
    query_x_to,
    query_y_from,
    query_y_to,
    time_from,
    time_to,
    output_crs,
    query_crs="EPSG:4326",
    dask_chunk_size="1500",
    **kwargs,
):

    baseline_query, baseline_water_product = _query(
        baseline_product,
        query_x_from,
        query_x_to,
        query_y_from,
        query_y_to,
        time_from,
        time_to,
        output_crs,
        query_crs,
        dask_chunk_size,
    )

    analysis_query, analysis_water_product = _query(
        analysis_product,
        query_x_from,
        query_x_to,
        query_y_from,
        query_y_to,
        time_from,
        time_to,
        output_crs,
        query_crs,
        dask_chunk_size,
    )

    b_ds = dc.load(baseline_query)
    a_ds = dc.load(analysis_query)

    if (
        len(b_ds.dims) == 0
        or len(b_ds.data_vars) == 0
        or len(a_ds.dims) == 0
        or len(a_ds.data_vars) == 0
    ):
        return None

    water_scenes_baseline_query = baseline_query
    water_scenes_baseline_query["product"] = baseline_water_product
    water_scenes_baseline_query["measurements"] = ["water_classification"]

    water_scenes_baseline = dc.load(water_scenes_baseline_query)

    water_scenes_analysis_query = analysis_query
    water_scenes_analysis_query["product"] = analysis_water_product
    water_scenes_analysis_query["measurements"] = ["water_classification"]

    water_scenes_analysis = dc.load(water_scenes_analysis_query)

    b_mask = mask_good_quality(b_ds, baseline_product)
    a_mask = mask_good_quality(a_ds, analysis_product)

    b_ds = b_ds.where(b_mask)
    a_ds = a_ds.where(a_mask)

    mosaic_function = {
        "median": create_median_mosaic,
        "max_ndvi": create_max_ndvi_mosaic,
    }

    new_compositor = mosaic_function[mosaic_type]

    if mosaic_type == "median":
        baseline_composite = new_compositor(b_ds, clean_mask=b_mask)
        analysis_composite = new_compositor(a_ds, clean_mask=a_mask)
    else:
        baseline_composite = dask.delayed(new_compositor)(b_ds, clean_mask=b_mask)
        analysis_composite = dask.delayed(new_compositor)(a_ds, clean_mask=a_mask)

    water_classes_baseline = water_scenes_baseline.where(water_scenes_baseline >= 0)
    water_classes_analysis = water_scenes_analysis.where(water_scenes_analysis >= 0)

    water_composite_baseline = water_classes_baseline.water_classification.mean(
        dim="time"
    )
    water_composite_analysis = water_classes_analysis.water_classification.mean(
        dim="time"
    )

    baseline_composite = baseline_composite.where(
        (baseline_composite != np.nan) & (water_composite_baseline == 0)
    )
    analysis_composite = analysis_composite.where(
        (analysis_composite != np.nan) & (water_composite_analysis == 0)
    )

    ndvi_baseline_composite = NDVI(baseline_composite)
    ndvi_analysis_composite = NDVI(analysis_composite)

    ndvi_anomaly = ndvi_analysis_composite - ndvi_baseline_composite

    ndvi_anomaly = ndvi_anomaly.compute()

    return ndvi_anomaly

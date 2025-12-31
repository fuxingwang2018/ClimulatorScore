
def crop_latlon_box(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Crop an xarray Dataset to a lat-lon bounding box.
    Supports both 1D and 2D lat/lon coordinates.

    Parameters:
        ds       : xarray.Dataset
        lat_min  : float
        lat_max  : float
        lon_min  : float
        lon_max  : float

    Returns:
        Cropped xarray.Dataset
    """

    # Handle 1D lat/lon
    if "lat" in ds.coords and "lon" in ds.coords:
        if ds["lat"].ndim == 1 and ds["lon"].ndim == 1:
            return ds.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )

    # Handle 2D lat/lon
    if "lat" in ds and "lon" in ds and ds["lat"].ndim == 2 and ds["lon"].ndim == 2:
        mask = (
            (ds["lat"] >= lat_min) & (ds["lat"] <= lat_max) &
            (ds["lon"] >= lon_min) & (ds["lon"] <= lon_max)
        )
    return ds.where(mask, drop=True)




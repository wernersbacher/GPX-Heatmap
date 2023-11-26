import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np


def deg2xy(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    """Returns OSM coordinates (x,y) from (lat,lon) in degree"""

    # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n

    return x, y


def xy2deg(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Returns (lat, lon) in degree from OSM coordinates (x,y)"""

    # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n)))
    lat_deg = np.degrees(lat_rad)

    return lat_deg, lon_deg


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Returns image filtered with a gaussian function of variance sigma**2"""

    i, j = np.meshgrid(np.arange(image.shape[0]),
                       np.arange(image.shape[1]),
                       indexing='ij')

    mu = (int(image.shape[0] / 2.0),
          int(image.shape[1] / 2.0))

    gaussian = 1.0 / (2.0 * np.pi * sigma * sigma) * np.exp(-0.5 * (((i - mu[0]) / sigma) ** 2 + ((j -
                                                                                                   mu[1]) / sigma) ** 2))

    gaussian = np.roll(gaussian, (-mu[0], -mu[1]), axis=(0, 1))

    image_fft = np.fft.rfft2(image)
    gaussian_fft = np.fft.rfft2(gaussian)

    image = np.fft.irfft2(image_fft * gaussian_fft)

    return image


def download_tile(tile_url: str, tile_file: str) -> bool:
    """Download tile from url to file, wait 0.1s and return True (False) if (not) successful"""

    request = Request(tile_url, headers={'User-Agent': 'GPX-Heatmap/1.0'})

    try:
        with urlopen(request) as response:
            data = response.read()

    except URLError as e:
        print(e)
        return False

    with open(tile_file, 'wb') as file:
        file.write(data)

    time.sleep(0.1)

    return True


class Points:
    files_used = 0
    data = []


def read_data_from_gpx(gpx_files: list[str], year_filter: int) -> Points:
    """Reads the data from gpx files and tracks used file count"""

    points = Points()

    for gpx_file in gpx_files:
        print('Reading {}'.format(os.path.basename(gpx_file)))
        file_used_in_points = 0

        with open(gpx_file, encoding='utf-8') as file:
            for line in file:
                if '<time' in line:
                    l = line.split('>')[1][:4]

                    if not year_filter or l in year_filter:
                        if not file_used_in_points:
                            file_used_in_points = 1
                            points.files_used += 1

                        for line in file:
                            if '<trkpt' in line:
                                l = line.split('"')

                                points.data.append([float(l[1]),
                                                    float(l[3])])
                    else:
                        break

    return points

import ee
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

"""
Climate Change Seasonal Temperature Comparison Tool

Users must authenticate Google Earth Engine before running:

    earthengine authenticate

Then run this script.

Default countries comparison but you can choose any countries and years you like:
- North: Norway, Sweden, Finland
- South: Spain, Portugal
- Years: 1990 and 2025
"""

# =========================================================
# USER SETTINGS
# =========================================================

PROJECT_ID = input("Enter your Google Earth Engine project ID: ").strip()

north_names = ["Norway", "Sweden", "Finland"]
south_names = ["Spain", "Portugal"]

years = [1990, 2025]

north_label = "NORTH"
south_label = "SOUTH"

vmin, vmax = -15, 30
palette = ["blue", "cyan", "lime", "yellow", "orange", "red"]


# =========================================================
# INITIALIZE GOOGLE EARTH ENGINE
# =========================================================

ee.Initialize(project=PROJECT_ID)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def winter_dates(year):
    return f"{year - 1}-12-01", f"{year}-03-01"


def summer_dates(year):
    return f"{year}-06-01", f"{year}-09-01"


def make_temp_map(start, end, region_fc, borders, vmin, vmax, palette):
    temp = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .select("temperature_2m")
        .mean()
        .subtract(273.15)
        .clip(region_fc)
    )

    white_bg = ee.Image.rgb(255, 255, 255)

    return (
        white_bg
        .blend(temp.visualize(min=vmin, max=vmax, palette=palette))
        .blend(borders)
    )


def ee_to_pil(image, region_geom):
    url = image.getThumbURL({
        "region": region_geom.bounds(),
        "dimensions": 1000,
        "format": "png"
    })

    response = requests.get(url)
    response.raise_for_status()

    return Image.open(BytesIO(response.content))


def make_group(countries_fc, country_names):
    return countries_fc.filter(
        ee.Filter.inList("country_na", country_names)
    )


def add_side_label(ax, text, color):
    ax.axis("off")

    box = FancyBboxPatch(
        (0.08, 0.30),
        0.84,
        0.40,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color + "12",
        transform=ax.transAxes
    )

    ax.add_patch(box)

    ax.text(
        0.5,
        0.5,
        text,
        fontsize=14,
        fontweight="bold",
        color=color,
        ha="center",
        va="center"
    )


# =========================================================
# LOAD COUNTRY BOUNDARIES
# =========================================================

countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

north = make_group(countries, north_names)
south = make_group(countries, south_names)

north_geom = north.geometry()
south_geom = south.geometry()

north_borders = ee.Image().paint(north, 1, 1).visualize(palette=["404040"])
south_borders = ee.Image().paint(south, 1, 1).visualize(palette=["404040"])


# =========================================================
# GENERATE MAP IMAGES
# =========================================================

north_winter, north_summer = [], []
south_winter, south_summer = [], []

for year in years:
    winter_start, winter_end = winter_dates(year)
    summer_start, summer_end = summer_dates(year)

    north_winter.append(
        ee_to_pil(
            make_temp_map(
                winter_start,
                winter_end,
                north,
                north_borders,
                vmin,
                vmax,
                palette
            ),
            north_geom
        )
    )

    north_summer.append(
        ee_to_pil(
            make_temp_map(
                summer_start,
                summer_end,
                north,
                north_borders,
                vmin,
                vmax,
                palette
            ),
            north_geom
        )
    )

    south_winter.append(
        ee_to_pil(
            make_temp_map(
                winter_start,
                winter_end,
                south,
                south_borders,
                vmin,
                vmax,
                palette
            ),
            south_geom
        )
    )

    south_summer.append(
        ee_to_pil(
            make_temp_map(
                summer_start,
                summer_end,
                south,
                south_borders,
                vmin,
                vmax,
                palette
            ),
            south_geom
        )
    )


# =========================================================
# PLOT FIGURE
# =========================================================

fig = plt.figure(figsize=(18, 9), facecolor="white")

gs = GridSpec(
    2,
    5,
    width_ratios=[0.55, 1, 1, 1, 1],
    hspace=0.02,
    wspace=0.03
)

year_1, year_2 = years

fig.suptitle(
    f"{north_label}–{south_label} Comparison of Mean Seasonal Temperature",
    fontsize=17,
    y=0.99
)

fig.text(
    0.37,
    0.91,
    f"{north_label} ({', '.join(north_names)})",
    fontsize=15,
    fontweight="bold",
    color="#2166ac",
    ha="center"
)

fig.text(
    0.75,
    0.91,
    f"{south_label} ({', '.join(south_names)})",
    fontsize=15,
    fontweight="bold",
    color="#e66101",
    ha="center"
)

fig.add_artist(Line2D([0.22, 0.55], [0.895, 0.895], color="#2166ac", linewidth=2))
fig.add_artist(Line2D([0.57, 0.90], [0.895, 0.895], color="#e66101", linewidth=2))

fig.add_artist(
    Line2D(
        [0.560, 0.560],
        [0.12, 0.875],
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6
    )
)

add_side_label(fig.add_subplot(gs[0, 0]), "WINTER", "#2166ac")
add_side_label(fig.add_subplot(gs[1, 0]), "SUMMER", "#e66101")

plot_data = [
    [north_winter[0], north_winter[1], south_winter[0], south_winter[1]],
    [north_summer[0], north_summer[1], south_summer[0], south_summer[1]]
]

title_colors = ["#2166ac", "#2166ac", "#e66101", "#e66101"]
column_titles = [str(year_1), str(year_2), str(year_1), str(year_2)]

for row in range(2):
    for col in range(4):
        ax = fig.add_subplot(gs[row, col + 1])
        ax.imshow(plot_data[row][col])
        ax.axis("off")
        ax.set_title(
            column_titles[col],
            fontsize=13,
            fontweight="bold",
            color=title_colors[col],
            pad=2
        )

cmap = mpl.colors.LinearSegmentedColormap.from_list("temp", palette)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([0.91, 0.20, 0.018, 0.62])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
cbar.set_label("Temperature (°C)", fontsize=11)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_linewidth(0.8)

plt.tight_layout()
plt.show()
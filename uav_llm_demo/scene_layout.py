"""Shared metric geometry for the browser and Unreal scene."""

from .scenarios import ScenarioDefinition
from pathlib import Path
import struct
import tempfile
import zlib

METERS_PER_CELL = 3.0
METERS_PER_LEVEL = 5.0
AIRSIM_ORIGIN = (500.0, 500.0, -100.0)


def color_texture(color: str) -> str:
    folder = Path(tempfile.gettempdir()) / "uav-demo-materials"
    folder.mkdir(exist_ok=True)
    path = folder / (color.lstrip("#") + ".png")
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    rgb = bytes.fromhex(color.lstrip("#"))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + rgb * 4) * 4))
                     + chunk(b"IEND", b""))
    return str(path)


def scene_layout(scenario: ScenarioDefinition) -> list[dict]:
    half = (scenario.grid_size - 1) / 2
    objects = []

    def box(name, x, y, z, width, depth, height, color):
        objects.append(dict(name=name, position=dict(x=x, y=y, z=z),
                            size=dict(x=width, y=depth, z=height), color=color))

    campus = scenario.mission_mode == "semantic"
    extent = 180 if scenario.urban_parcels and not campus else scenario.grid_size * METERS_PER_CELL + 22
    box("ground", half, half, -0.1, extent, extent, 1,
        "#8d9c80" if campus else "#a9b0a9" if scenario.urban_parcels else "#75945e")
    if campus:
        layout = scenario.campus_layout or {}
        # The detailed browser campus and this conservative Unreal export read
        # the same source. Ground markings never introduce flight obstacles.
        for item in layout.get("surfaces", []):
            left, right, north, south = item["footprint"]
            box("campus_" + item["id"].replace("-", "_"),
                (left + right) / 2, (north + south) / 2, -0.006,
                (right - left + 0.78) * METERS_PER_CELL,
                (south - north + 0.78) * METERS_PER_CELL, 0.04, item["color"])
        for kind, color in (("waterways", "#789f9d"), ("roads", "#687271")):
            for item in layout.get(kind, []):
                for index, ((x1, y1), (x2, y2)) in enumerate(zip(item["points"], item["points"][1:])):
                    # The box-only export approximates diagonal ground strips;
                    # they are decorative and lie beneath the flight floor.
                    box(f"campus_{item['id']}_{index}", (x1 + x2) / 2, (y1 + y2) / 2,
                        -0.004 if kind == "roads" else -0.016,
                        (abs(x2 - x1) + item["width"]) * METERS_PER_CELL,
                        (abs(y2 - y1) + item["width"]) * METERS_PER_CELL,
                        0.03, color)
    if scenario.urban_parcels:
        centers = scenario.urban_street_centers
        service_centers = tuple((before + after) / 2 for before, after in zip(centers, centers[1:]))
        crossings = sorted([(center, 1.6) for center in centers]
                           + [(center, 0.9) for center in service_centers])
        sidewalk_segments = []
        cursor = half - (extent - 6) / (2 * METERS_PER_CELL)
        for center, radius in crossings:
            sidewalk_segments.append((cursor, center - radius))
            cursor = center + radius
        sidewalk_segments.append((cursor, half + (extent - 6) / (2 * METERS_PER_CELL)))
        for index, center in enumerate(centers):
            suffix = "" if index == 0 else f"_{index}"
            box("road_north_south" + suffix, center, half, 0.005, 9.6, extent - 6, 0.05, "#39454b")
            box("road_east_west" + suffix, half, center, 0.006, extent - 6, 9.6, 0.05, "#39454b")
            for side, offset in enumerate((-5.5, 5.5)):
                edge = center + offset / METERS_PER_CELL
                for segment, (start, end) in enumerate(sidewalk_segments):
                    middle = (start + end) / 2
                    length = (end - start) * METERS_PER_CELL
                    box(f"sidewalk_ns_{index}_{side}_{segment}", edge, middle, 0.025,
                        1.6, length, 0.12, "#cbd0c8")
                    box(f"sidewalk_ew_{index}_{side}_{segment}", middle, edge, 0.025,
                        length, 1.6, 0.12, "#cbd0c8")
            for stripe, cell in enumerate(range(2, scenario.grid_size - 1, 3)):
                if any(abs(cell - crossing) < 3 for crossing, _ in crossings):
                    continue
                box(f"lane_ns_{index}_{stripe}", center, cell, 0.016, 0.13, 2.6, 0.02, "#dfddc9")
                box(f"lane_ew_{index}_{stripe}", cell, center, 0.016, 2.6, 0.13, 0.02, "#dfddc9")
        for ix, east in enumerate(centers):
            for iy, south in enumerate(centers):
                for side, offset in enumerate((-1.75, 1.75)):
                    for stripe in range(6):
                        across = (-3.75 + stripe * 1.5) / METERS_PER_CELL
                        box(f"cross_ns_{ix}_{iy}_{side}_{stripe}", east + across, south + offset,
                            0.018, 0.72, 1.9, 0.02, "#e5e4d8")
                        box(f"cross_ew_{ix}_{iy}_{side}_{stripe}", east + offset, south + across,
                            0.018, 1.9, 0.72, 0.02, "#e5e4d8")
        for index, middle in enumerate(service_centers):
            box(f"road_service_ns_{index}", middle, half, 0.006, 5.4, extent - 6, 0.05, "#39454b")
            box(f"road_service_ew_{index}", half, middle, 0.007, extent - 6, 5.4, 0.05, "#39454b")
    elif not campus:
        for lane, offset in enumerate((-18, 18)):
            x = half + offset / METERS_PER_CELL
            box(f"road_{lane}", x, half, 0.005, 5.4, 78, 0.05, "#4b5150")
            for stripe, z in enumerate(range(-34, 35, 6)):
                box(f"stripe_{lane}_{stripe}", x, half + z / METERS_PER_CELL,
                    0.012, 0.15, 2.8, 0.02, "#e7dfb4")
    colors = ("#bcc4c3", "#9da9aa", "#c6b8a1", "#7e9195", "#b2aca2")
    for index, (x, y, height) in enumerate(scenario.buildings):
        box(f"building_{index}", x, y, height / 2,
            METERS_PER_CELL * 0.78, METERS_PER_CELL * 0.78,
            height * METERS_PER_LEVEL, colors[index % len(colors)])
    for name, point, color in (("base", scenario.base, "#4fd477"),
                               ("target", scenario.target, "#f2b84b")):
        # The ground launch marker must not become a raised collision block.
        pad_height = -0.014 if name == "base" and point[2] == 0 else 0.025
        box(name + "_pad", point[0], point[1], pad_height, 2.5, 2.5, 0.15, color)
    return objects

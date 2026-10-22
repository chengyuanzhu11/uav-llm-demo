"""Reference-based Hangzhou Beihang campus layout in the simulator grid.

This is a proportioned interpretation of the supplied aerial photo and guide map,
not surveyed coordinates. North is decreasing y. One horizontal cell is 3 metres;
one height level is 5 metres. Inclusive integer footprints use the existing
collision-cell outer edge (0.39 cells). Curved roofs reserve their entire maximum
height column, including glazed skylights. Roof apertures never imply a safe
underpass; only explicitly listed, completely open holes can be omitted.
"""
from __future__ import annotations

import math

from .path_tracking import PathFollower
from .safety import SAFETY_MARGIN, SCALE, _intersects


def _building(identifier, label, kind, footprint, height, facade="glass"):
    return dict(id=identifier, label=label, kind=kind, footprint=list(footprint),
                height=height, facade=facade)


def _surface(identifier, kind, footprint, color):
    return dict(id=identifier, kind=kind, footprint=list(footprint), color=color)


def _line(identifier, points, width):
    return dict(id=identifier, points=[list(point) for point in points], width=width)


def _research(identifier, label, x, y, mirrored=False):
    # Three perimeter wings and a shorter central wing leave a real open court.
    west, east = x, x + 12
    side = (east - 2, east) if mirrored else (west, west + 2)
    middle = (west + 4, east) if mirrored else (west, east - 4)
    return [
        _building(identifier + "-north", label, "research", (west, east, y, y + 2), 2.5),
        _building(identifier + "-spine", label, "research", (*side, y + 3, y + 12), 2.5),
        _building(identifier + "-middle", label, "research", (*middle, y + 7, y + 9), 2.5),
        _building(identifier + "-south", label, "research", (west, east, y + 13, y + 15), 2.5),
    ]


_BUILDINGS = [
    _building("C1", "会议中心 C1", "conference", (41, 46, 64, 70), 2.6, "stone"),
    _building("C2", "图书馆 C2", "library", (52, 56, 64, 70), 2.6),
    _building("C3", "教学一号楼 C3", "teaching", (41, 46, 57, 60), 2.4),
    _building("C4", "教学二号楼 C4", "teaching", (52, 56, 57, 60), 2.4),
    _building("C5", "科教美育中心 C5", "exhibition", (46, 52, 48, 50), 2.0, "stone"),
    _building("C6", "教学三号楼 C6", "teaching", (41, 46, 40, 44), 2.4),
    _building("C7", "教学四号楼 C7", "teaching", (52, 56, 40, 44), 2.4),
    _building("C8", "音乐厅 C8", "auditorium", (41, 46, 30, 35), 2.6, "stone"),
    _building("C9", "教学五号楼 C9", "teaching", (52, 56, 30, 35), 2.6),
    *_research("R1", "科研一号楼 R1", 22, 58),
    *_research("R2", "科研二号楼 R2", 22, 29),
    *_research("R3", "科研三号楼 R3", 64, 29, mirrored=True),
    _building("L1", "实验一号楼 L1", "laboratory", (65, 77, 87, 89), 2.9),
    _building("L2", "实验二号楼 L2", "laboratory", (65, 67, 79, 86), 2.9),
    _building("L3", "实验三号楼 L3", "laboratory", (75, 77, 83, 86), 2.9),
    _building("L4", "实验四号楼 L4", "laboratory", (68, 77, 79, 81), 2.9),
    _building("L5", "实验五号楼 L5", "laboratory", (31, 35, 18, 23), 2.1, "stone"),
    _building("L6", "实验六号楼 L6", "laboratory", (31, 35, 10, 16), 2.1, "stone"),
    _building("north-lab-wing", "北区实验楼", "laboratory", (22, 28, 10, 12), 1.9, "stone"),
    _building("S2-S4", "南区体育馆 S2–S4", "sports_hall", (31, 36, 79, 89), 1.9),
    _building("south-canteen", "南区食堂", "canteen", (62, 67, 70, 73), 1.5, "stone"),
    _building("north-canteen", "北区食堂", "canteen", (51, 57, 22, 25), 1.5, "stone"),
]
for identifier, footprint in (
    ("D1", (72, 77, 71, 73)), ("D2", (62, 69, 65, 67)),
    ("D3", (72, 77, 65, 67)), ("D4", (62, 69, 60, 62)),
    ("D5", (72, 77, 60, 62)), ("D6", (62, 69, 56, 58)),
    ("D7", (72, 77, 56, 58)), ("D8", (60, 68, 23, 25)),
    ("D9", (51, 57, 17, 19)), ("D10", (60, 68, 17, 19)),
    ("D11", (51, 57, 12, 14)), ("D12", (60, 68, 12, 14)),
    ("D13", (51, 57, 7, 9)), ("D14", (60, 68, 7, 9)),
):
    _BUILDINGS.append(_building(identifier, "学生宿舍 " + identifier, "residential", footprint,
                                2.3, "bronze" if int(identifier[1:]) % 3 == 1 else "glass"))
for identifier, footprint in (
    ("A1", (71, 77, 23, 25)), ("A2", (71, 77, 17, 19)),
    ("A3", (71, 77, 12, 14)), ("A4", (71, 77, 7, 9)),
    ("A5", (77, 78, 15, 16)),
):
    _BUILDINGS.append(_building(identifier, "教师发展中心 A5" if identifier == "A5" else "教师公寓 " + identifier,
                                "faculty", footprint, 2.1, "stone"))


CAMPUS_LAYOUT = {
    "name": "杭州市北京航空航天大学国际创新研究院",
    "north": "negative_y",
    "buildings": _BUILDINGS,
    "roofs": [
        dict(id="central-south-roof", label="中心组团南区", type="curved", axis="y", curve_range=[29, 71],
             footprint=[40, 57, 56, 71], height=3.6, sag=0.8, holes=[],
             skylights=[[42, 44, 58, 60], [47, 50, 60, 63], [53, 55, 58, 60],
                        [42, 44, 66, 68], [53, 55, 66, 68]]),
        dict(id="central-north-roof", label="中心组团北区", type="curved", axis="y", curve_range=[29, 71],
             footprint=[40, 57, 29, 45], height=3.6, sag=0.8, holes=[],
             skylights=[[42, 44, 31, 33], [47, 50, 35, 39], [53, 55, 31, 33],
                        [42, 44, 40, 42], [53, 55, 40, 42]]),
        dict(id="central-link-roof", label="中心景观廊", type="ribbed",
             axis="y", curve_range=[29, 71], footprint=[40, 57, 46, 55],
             height=2.9, sag=0.1, holes=[], skylights=[]),
        dict(id="sports-hall-roof", label="南区体育馆", type="wave",
             footprint=[31, 36, 79, 89], height=2.35, sag=0.4, holes=[]),
    ],
    "surfaces": [
        _surface("south-plaza", "plaza", (40, 57, 78, 86), "#c9ccc3"),
        _surface("central-garden", "garden", (39, 58, 46, 54), "#8eac7c"),
        _surface("west-garden", "garden", (21, 35, 46, 54), "#87a886"),
        _surface("east-garden", "garden", (62, 77, 46, 54), "#8ba78b"),
        _surface("south-court", "paving", (39, 58, 55, 74), "#c3c5bd"),
        _surface("north-court", "paving", (39, 58, 28, 46), "#c3c5bd"),
        _surface("south-track", "running_track", (20, 29, 78, 89), "#b6584a"),
        _surface("north-football", "football", (40, 44, 20, 25), "#79a789"),
        _surface("north-volleyball", "volleyball", (45, 48, 22, 25), "#508d9f"),
        _surface("north-tennis", "tennis", (45, 48, 18, 21), "#769c89"),
        _surface("north-basketball", "basketball", (40, 48, 10, 17), "#4c96ad"),
        _surface("research-one-court", "paving", (22, 34, 58, 73), "#bcbfb6"),
        _surface("research-two-court", "paving", (22, 34, 29, 44), "#bcbfb6"),
        _surface("research-three-court", "paving", (64, 76, 29, 44), "#bcbfb6"),
        _surface("laboratory-court", "paving", (65, 77, 79, 89), "#bfc3bc"),
        _surface("launch-apron", "launch", (47, 49, 87, 89), "#8a9391"),
    ],
    "roads": [
        _line("campus-ring", [(19, 8), (38, 8), (38, 6), (80, 6), (80, 90), (18, 90), (18, 27), (19, 8)], 1.2),
        _line("campus-west-road", [(38, 9), (38, 90)], 1.5),
        _line("campus-east-road", [(59, 6), (59, 90)], 1.2),
        _line("campus-north-road", [(19, 27), (82, 27)], 1.6),
        _line("campus-south-road", [(18, 76), (80, 76)], 1.8),
        _line("campus-garden-path", [(18, 55), (80, 55)], 0.8),
        _line("south-entry", [(48, 95), (48, 89)], 3.0),
        _line("south-east-entry", [(60, 95), (60, 90)], 1.4),
        _line("north-entry", [(47, 0), (47, 7)], 1.5),
        _line("south-city-road", [(4, 94.5), (91, 94.5)], 2.0),
        _line("east-city-road", [(87, 2), (87, 93)], 2.0),
        _line("west-city-road", [(11, 2), (11, 93)], 2.0),
        _line("north-city-road", [(11, 1), (87, 1)], 2.0),
    ],
    "waterways": [
        _line("campus-moat", [(16, 3.9), (82, 3.9), (82, 92), (16, 92), (16, 3.9)], 2.6),
        _line("west-garden-stream", [(16, 47), (26, 47), (30, 51), (23, 54), (16, 54)], 1.3),
        _line("east-garden-stream", [(82, 47), (71, 47), (65, 50), (72, 54), (82, 54)], 1.3),
        _line("southern-river", [(4, 97), (92, 97)], 3.4),
    ],
    # Small ornamental trees are below the 10 m flight floor. No trees grow on
    # the launch apron, roads, sports grounds, observation points or roof holes.
    "trees": [
        dict(id=f"tree-{index}", x=x, y=y, height=0.5, radius=0.38)
        for index, (x, y) in enumerate(
            [(x, y) for x in (20, 36, 61, 79) for y in range(10, 90, 4)
             if y not in (26, 46, 54, 74, 78, 82, 86)]
            + [(x, y) for x in (42, 45, 52, 55) for y in (80, 84, 87)]
        )
    ],
}


def _campus_ground_obstacles(layout):
    """Low solid structures shared by rendering and ground-transition safety.

    Bounds are exact GRID boxes rather than collision-cell footprints. Geometry
    that used to live only in the renderer belongs here so a new surface target
    cannot place a landing column through a decorative but solid structure.
    """
    obstacles = []

    def box(identifier, kind, center, size_m, material):
        radius = [size / scale / 2 for size, scale in zip(size_m, SCALE)]
        obstacles.append(dict(id=identifier, kind=kind, material=material,
                              lower_grid=[value - delta for value, delta in zip(center, radius)],
                              upper_grid=[value + delta for value, delta in zip(center, radius)]))

    for surface in layout["surfaces"]:
        if surface["kind"] != "running_track":
            continue
        left, right, north, south = surface["footprint"]
        depth_m = (south - north + .78) * SCALE[1]
        center_y = (north + south) / 2
        for step in range(4):
            box(f"campus_{surface['id']}_stand_step_{step}", "spectator_stand",
                (left - .39 + (.22 + step * .4) / SCALE[0], center_y,
                 (.25 + (3 - step) * .19) / SCALE[2]),
                (.4, depth_m * .57, .22 + (3 - step) * .38), "stone")
        box(f"campus_{surface['id']}_stand_canopy", "spectator_canopy",
            (left - .39 + 1.15 / SCALE[0], center_y, 2.55 / SCALE[2]),
            (2.3, depth_m * .61, .06), "paleMetal")
    entry = next((road for road in layout["roads"] if road["id"] == "south-entry"), None)
    if entry:
        entry_x = entry["points"][0][0]
        entry_y = sum(point[1] for point in entry["points"]) / len(entry["points"])
        for side in (-1, 1):
            box(f"campus_south_gate_pier_{side}", "gate_pier",
                (entry_x + side * 3.3 / SCALE[0], entry_y - .5, 1.24 / SCALE[2]),
                (.62, .72, 2.4), "trim")
            box(f"campus_south_gate_rail_{side}", "bridge_rail",
                (entry_x + side * 3.9 / SCALE[0], entry_y, .64 / SCALE[2]),
                (.16, 10.2, .75), "paleMetal")
        box("campus_south_gate_sign", "monument_sign",
            (entry_x + 6, entry_y - 1.3, 1.25 / SCALE[2]),
            (9.6, .4, 2.35), "stone")
    return obstacles


CAMPUS_LAYOUT["ground_obstacles"] = _campus_ground_obstacles(CAMPUS_LAYOUT)


def _track_geometry(footprint):
    """One metric capsule centerline shared by rendering and motion compilation."""
    left, right, north, south = footprint
    cx, cy = (left + right) / 2, (north + south) / 2
    radius = (right - left) / 2 - 1.2
    half = (south - north) / 2 - 1.2 - radius
    points = [(cx + radius, cy - half), (cx + radius, cy + half)]
    for index in range(1, 49):
        angle = math.pi * index / 48
        points.append((cx + radius * math.cos(angle), cy + half + radius * math.sin(angle)))
    points.append((cx - radius, cy - half))
    for index in range(1, 49):
        angle = math.pi + math.pi * index / 48
        points.append((cx + radius * math.cos(angle), cy - half + radius * math.sin(angle)))
    points[-1] = points[0]
    return dict(kind="capsule", center_grid=[cx, cy], radius_grid=radius,
                straight_half_grid=half, centerline_grid=[list(point) for point in points],
                lane_width_m=2.4)


def _bounds(items):
    return (min(item["footprint"][0] for item in items),
            max(item["footprint"][1] for item in items),
            min(item["footprint"][2] for item in items),
            max(item["footprint"][3] for item in items))


def _position(x, y, z):
    return dict(x=x, y=y, z=z)


def _label_position(layout, x, y, height):
    # Labels must also clear shared canopies, not only individual building roofs.
    for roof in layout["roofs"]:
        left, right, north, south = roof["footprint"]
        if left - .39 <= x <= right + .39 and north - .39 <= y <= south + .39:
            height = max(height, roof["height"])
    return _position(x, y, height + .35)


def _outdoor_arrival(layout, footprint, *, building=False):
    """Choose a stable 10 m arrival outside every solid roof/building envelope."""
    left, right, north, south = footprint
    cx, cy = (left + right) / 2, (north + south) / 2
    clearance = SAFETY_MARGIN + .5
    margin = .39 + clearance / SCALE[0]
    obstacles = [item for item in [*layout["buildings"], *layout["roofs"]]
                 if item["height"] * SCALE[2] + clearance >= 10]

    def clear(x, y):
        if not (1 <= x <= 94 and 1 <= y <= 94):
            return False
        for item in obstacles:
            l, r, n, s = item["footprint"]
            if l - margin <= x <= r + margin and n - margin <= y <= s + margin:
                return False
        return True

    candidates = []
    # Model centres describe a place; arrival points are independently chosen in
    # open air. A whole R-group court is treated as one building exterior here.
    for xi in range(math.floor((left - 8) * 2), math.ceil((right + 8) * 2) + 1):
        for yi in range(math.floor((north - 8) * 2), math.ceil((south + 8) * 2) + 1):
            x, y = xi / 2, yi / 2
            if building and left - margin <= x <= right + margin and north - margin <= y <= south + margin:
                continue
            if clear(x, y):
                # Prefer the nearest exterior and, for ties, the south frontage.
                candidates.append(((x - cx) ** 2 + (y - cy) ** 2, -y, x, y))
    if not candidates:
        raise ValueError(f"No clear campus arrival for footprint {footprint}.")
    _, _, x, y = min(candidates)
    return _position(x, y, 2)


def _campus_places(layout):
    """The sole semantic mapping from physical model parts to named places."""
    actions = ["visit", "inspect", "capture"]
    # Persisted task IDs and established observation positions remain unchanged.
    established = {
        "C1": ("conference", (43, 74, 3)),
        "C2": ("library", (53, 74, 3)),
        "C3": ("teaching", (38, 54, 2.8)),
        "C5": ("exhibition", (60, 52, 2.4)),
        "R1": ("research_one", (36, 69, 2.5)),
        "R2": ("research_two", (36, 35, 2.5)),
        "L1": ("laboratory", (62, 84, 3.5)),
        "C9": ("north_teaching", (53, 27, 3)),
    }
    groups = {}
    for item in layout["buildings"]:
        identifier = item["id"].split("-")[0] if item["kind"] == "research" else item["id"]
        groups.setdefault(identifier, []).append(item)
    research_surfaces = {"R1": "research-one-court", "R2": "research-two-court", "R3": "research-three-court"}
    places = []
    for identifier, items in groups.items():
        footprint = _bounds(items)
        left, right, north, south = footprint
        item = items[0]
        site_id, fixed = established.get(identifier, (identifier.lower().replace("-", "_"), None))
        code = identifier if identifier[0] in "CDALRS" and identifier[1].isdigit() else item["label"]
        permitted = list(actions)
        if site_id in {"library", "laboratory"}:
            permitted.insert(2, "thermal_scan")
        places.append(dict(
            id=site_id, label=item["label"], code=code, kind=item["kind"],
            building_ids=[part["id"] for part in items],
            surface_ids=[research_surfaces[identifier]] if identifier in research_surfaces else [],
            position=_position(*fixed) if fixed else _outdoor_arrival(layout, footprint, building=True),
            label_position=_label_position(layout, (left + right) / 2, (north + south) / 2,
                                           max(part["height"] for part in items)),
            allowed_actions=permitted))

    surface_names = {
        "south-plaza": "南广场", "central-garden": "中心花园", "west-garden": "西侧花园",
        "east-garden": "东侧花园", "south-court": "中心南庭院", "north-court": "中心北庭院",
        "north-football": "北区足球场", "north-volleyball": "北区排球场",
        "north-tennis": "北区网球场", "north-basketball": "北区篮球场",
        "laboratory-court": "实验楼庭院",
    }
    for surface in layout["surfaces"]:
        identifier = surface["id"]
        if identifier in research_surfaces.values():
            continue
        left, right, north, south = surface["footprint"]
        if identifier == "launch-apron":
            site_id, label, code, kind = "home", "南门起降点", "南门起降点", "home"
            position, permitted = _position(48, 88, 0), ["return_home"]
        elif surface["kind"] == "running_track":
            site_id, label, code, kind = "stadium", "田径场 S1", "S1", "sports"
            position, permitted = _position(26, 84, 2.2), list(actions)
        else:
            site_id, label = identifier.replace("-", "_"), surface_names[identifier]
            code, kind = label, surface["kind"]
            position, permitted = _outdoor_arrival(layout, surface["footprint"]), list(actions)
        places.append(dict(id=site_id, label=label, code=code, kind=kind,
                           building_ids=[], surface_ids=[identifier], position=position,
                           label_position=_label_position(layout, (left + right) / 2,
                                                          (north + south) / 2, .15),
                           allowed_actions=permitted))
    for identifier, label, position in (
            ("east_gate", "东门", (81, 26, 2)),
            ("north_gate", "北门", (47, 6, 2))):
        places.append(dict(id=identifier, label=label, code=label, kind="gate",
                           building_ids=[], surface_ids=[], position=_position(*position),
                           label_position=_position(position[0], position[1], .7),
                           allowed_actions=list(actions)))
    return places


def _place_geometry(layout):
    buildings = {item["id"]: item for item in layout["buildings"]}
    surfaces = {item["id"]: item for item in layout["surfaces"]}
    objects = {}
    for surface in surfaces.values():
        if surface["kind"] == "running_track":
            surface["geometry"] = _track_geometry(surface["footprint"])
    for place in layout["places"]:
        items = [buildings[identifier] for identifier in place["building_ids"]]
        if not items:
            items = [surfaces[identifier] for identifier in place["surface_ids"]]
        if items:
            left, right, north, south = _bounds(items)
        else:
            x, y = place["position"]["x"], place["position"]["y"]
            left, right, north, south = x - .8, x + .8, y - .8, y + .8
        margin = .8 if place["building_ids"] else 0
        boundary = [[left - margin, north - margin], [right + margin, north - margin],
                    [right + margin, south + margin], [left - margin, south + margin],
                    [left - margin, north - margin]]
        if any(surfaces[identifier]["kind"] == "running_track" for identifier in place["surface_ids"]):
            boundary = surfaces[place["surface_ids"][0]]["geometry"]["centerline_grid"]
        objects[place["id"]] = dict(center_grid=[(left + right) / 2, (north + south) / 2],
                                     boundary_grid=boundary)
    clearance = SAFETY_MARGIN + PathFollower.CORRIDOR_M
    xy_margin = .39 + clearance / SCALE[0]
    for geometry in objects.values():
        roof_altitude = 0.0
        boundary = geometry["boundary_grid"]
        for item in [*layout["buildings"], *layout["roofs"]]:
            left, right, north, south = item["footprint"]
            lower = (left - xy_margin, north - xy_margin, -1.0)
            upper = (right + xy_margin, south + xy_margin, 1.0)
            if any(_intersects((*a, 0.0), (*b, 0.0), lower, upper)
                   for a, b in zip(boundary, boundary[1:])):
                roof_altitude = max(roof_altitude, item["height"] * SCALE[2])
        # Computed once from the same conservative envelopes as collision geometry.
        # Omitted heights can clear roofs; explicit user heights are never rewritten.
        geometry["boundary_min_altitude_m"] = roof_altitude + clearance + .1 if roof_altitude else 0.0
    return objects


CAMPUS_LAYOUT["places"] = _campus_places(CAMPUS_LAYOUT)
CAMPUS_LAYOUT["spatial_objects"] = _place_geometry(CAMPUS_LAYOUT)


def campus_building_cells(layout=CAMPUS_LAYOUT):
    """Build conservative obstacle columns from exactly the rendered geometry."""
    cells = {}
    for item in [*layout["buildings"], *layout["roofs"]]:
        left, right, north, south = item["footprint"]
        holes = item.get("holes", [])
        for x in range(left, right + 1):
            for y in range(north, south + 1):
                if any(l <= x <= r and n <= y <= s for l, r, n, s in holes):
                    continue
                cells[x, y] = max(cells.get((x, y), 0), item["height"])
    return tuple((x, y, height) for (x, y), height in sorted(cells.items()))

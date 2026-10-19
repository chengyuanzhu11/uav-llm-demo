"""Reference-based Hangzhou Beihang campus layout in the simulator grid.

This is a proportioned interpretation of the supplied aerial photo and guide map,
not surveyed coordinates. North is decreasing y. One horizontal cell is 3 metres;
one height level is 5 metres. Inclusive integer footprints use the existing
collision-cell outer edge (0.39 cells). Curved roofs reserve their entire maximum
height column, including glazed skylights. Roof apertures never imply a safe
underpass; only explicitly listed, completely open holes can be omitted.
"""
from __future__ import annotations


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

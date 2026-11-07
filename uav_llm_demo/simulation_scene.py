"""Shared scene assets; identities stay in the renderer/evaluator, not VLM input."""
import json
from pathlib import Path

SCENES = Path(__file__).resolve().parents[1] / "simulation" / "scenes"


def load_scene(name):
    if not isinstance(name, str) or not name.replace('_', '').isalnum():
        raise ValueError("Invalid simulation scene name")
    return json.loads((SCENES / (name + '.json')).read_text())


def scene_boxes(scene):
    """Visual and collision components in project east/south/up metres."""
    parts = []
    def box(name, p, size, color, shape="box"):
        parts.append(dict(name=name, position=dict(zip('xyz', (p[0]/3, p[1]/3, p[2]/5))),
                          size=dict(zip('xyz', size)), color=color, shape=shape))
    for obj in scene['objects']:
        x,y,z = obj['position_m']; name=obj['id']; kind=obj['kind']
        if kind == 'box':
            box(name, (x,y,z), obj['size_m'], '#a69980')
        elif kind == 'tree':
            h=obj['height_m']; r=obj['radius_m']
            box(name+'_trunk', (x,y,z+h*.35), (.35,.35,h*.7), '#735034', 'cylinder')
            box(name+'_crown', (x,y,z+h*.75), (2*r,2*r,h*.5), '#397348', 'ellipsoid')
        elif kind == 'bin':
            w,d,h=obj['size_m']; box(name, (x,y,z+h/2), (w,d,h), '#09591f')
            box(name+'_lid', (x,y,z+h), (w+.1,d+.1,.12), '#17251b')
        elif kind == 'bench':
            box(name+'_seat', (x,y,z+.5), (2,.55,.15), '#804014')
            box(name+'_back', (x,y+.25,z+.9), (2,.12,.65), '#804014')
            for dx in (-.7,.7):box(name+str(dx), (x+dx,y,z+.2), (.1,.45,.4), '#303030')
    return parts


def safety_cells(scene):
    import math
    cells={}
    for obj in scene_boxes(scene):
        p,s=obj['position'],obj['size']
        for x in range(math.floor(p['x']-s['x']/6), math.ceil(p['x']+s['x']/6)+1):
            for y in range(math.floor(p['y']-s['y']/6), math.ceil(p['y']+s['y']/6)+1):
                cells[x,y]=max(cells.get((x,y),0), (p['z']*5+s['z']/2)/5)
    return tuple((x,y,h) for (x,y),h in cells.items())


def campus_collision_geometry(layout):
    """Conservative solids from the same envelopes used by campus flight safety.

    Visual meshes are exported from the existing renderer. These inexpensive
    boxes deliberately keep the planner's solid-column roof approximation;
    they do not claim to model traversable interiors or branches individually.
    """
    parts = []

    def box(name, lower, upper):
        parts.append(dict(name=name,
            position=dict(zip('xyz', ((a+b)/2 for a,b in zip(lower,upper)))),
            size=dict(zip('xyz', ((b-a)*s for a,b,s in zip(lower,upper,(3,3,5)))))))

    for item in [*layout['buildings'], *layout['roofs']]:
        left,right,north,south = item['footprint']
        holes = item.get('holes', [])
        if holes:
            for x in range(left,right+1):
                for y in range(north,south+1):
                    if not any(l<=x<=r and n<=y<=s for l,r,n,s in holes):
                        box(f"{item['id']}_{x}_{y}", (x-.39,y-.39,0), (x+.39,y+.39,item['height']))
        else:
            box(item['id'], (left-.39,north-.39,0), (right+.39,south+.39,item['height']))
    for item in layout['ground_obstacles']:
        box(item['id'], item['lower_grid'], item['upper_grid'])
    for tree in layout['trees']:
        x,y,r = tree['x'],tree['y'],tree['radius']
        box(tree['id'], (x-r,y-r,0), (x+r,y+r,tree['height']))
    return parts

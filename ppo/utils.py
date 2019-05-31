import random

TRACKS = [
    ("e-track-1", "road"),
    ("e-track-2", "road"),
    ("e-track-3", "road"),
    ("e-track-4", "road"),
    ("e-track-6", "road"),
]

def _find_by_name(nodes, key):
    for node in nodes:
        if node.attrib["name"] == key:
            return node

def _find_by_tag(nodes, tag):
    for node in nodes:
        if node.tag == tag:
            return node

def sample_track(root_node):
    node = _find_by_name(root_node, "Tracks")
    subnode = _find_by_tag(node, "section")
    trackname_node, tracktype_node = subnode.getchildren()
    
    trackname, tracktype = random.sample(TRACKS, 1)[0]

    trackname_node.attrib["val"] = trackname
    tracktype_node.attrib["val"] = tracktype

def set_render_mode(root_node, render=True):
    node = _find_by_name(root_node, "Quick Race")
    subnode = _find_by_name(node, "display mode")
    if render:
        subnode.attrib["val"] = "normal"
    else:
        subnode.attrib["val"] = "results only"


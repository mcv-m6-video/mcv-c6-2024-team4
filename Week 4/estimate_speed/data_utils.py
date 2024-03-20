import xml.etree.ElementTree as elemTree
from typing import Dict
import os

# Code for parsing the XML annotation files from Team1-2023
def sort_dict(dictionary: Dict):
    """
    Sorts a dictionary by the key.
    :param dictionary: dictionary to sort
    :return: sorted dictionary
    """
    frames_num_str = list(dictionary.keys())
    frames_int = sorted(int(frame[2:]) for frame in frames_num_str)
    return {i: dictionary["f_" + str(i)] for i in frames_int}

def parse_pascalvoc_annotations(path_to_xml_file, add_track_id=False, removed_parked=False):
    """
    Parses an XML annotation file in the Pascal VOC format and extracts bounding box coordinates for cars in each frame.
    Args:
    - path_to_xml_file: path to the annotation XML file.

    Returns:
    - A dictionary of frame numbers and their corresponding car bounding box coordinates.
        example: dict = {'f_0': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]}
        where x1, y1, x2, y2 are the coordinates of the bounding box in the top left and bottom right corners.

    """

    tree = elemTree.parse(path_to_xml_file)
    root = tree.getroot()

    # Initialize an empty dictionary to hold the frame numbers and their corresponding bounding box coordinates
    frame_dict = {}

    # Loop through each 'track' element in the XML file with a 'label' attribute of 'car'
    for track in root.findall(".//track[@label='car']"):
        track_id = int(track.attrib["id"])
        # Loop through each 'box' element within the 'track' element to get the bounding box coordinates
        for box, attribute in zip(track.findall(".//box"), track.findall(".//attribute")):

            if removed_parked and attribute.text == "true":
                continue
            else:
                # Extract the bounding box coordinates and the frame number
                x1 = float(box.attrib['xtl'])
                y1 = float(box.attrib['ytl'])
                x2 = float(box.attrib['xbr'])
                y2 = float(box.attrib['ybr'])
                frame_num = f"f_{box.attrib['frame']}"

                # If the frame number doesn't exist in the dictionary yet, add it and initialize an empty list
                if frame_num not in frame_dict:
                    frame_dict[frame_num] = []

                # Append the bounding box coordinates to the list for the current frame number
                if add_track_id:
                    frame_dict[frame_num].append([x1, y1, x2, y2, track_id])
                else:
                    frame_dict[frame_num].append([x1, y1, x2, y2])

    return sort_dict(frame_dict)


def parse_cvat_annotations(cvat_file):
    tree = elemTree.parse(cvat_file)
    root = tree.getroot()

    annotations = {}

    for track in root.findall('./track'):
        track_id = int(track.attrib['id'])
        
        for bbox in track.findall('./box'):
            frame = int(bbox.attrib['frame'])
            xtl = float(bbox.attrib['xtl'])
            ytl = float(bbox.attrib['ytl'])
            xbr = float(bbox.attrib['xbr'])
            ybr = float(bbox.attrib['ybr'])
            box = [xtl, ytl, xbr, ybr]
            
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append([xtl, ytl, xbr, ybr, track_id])

    return annotations






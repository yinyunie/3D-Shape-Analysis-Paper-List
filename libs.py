# Added a new paper to the list.
# Author: Yinyu Nie
# Updated data: 23 June, 2021
import argparse

existing_topics = ['3D Detection & Segmentation',
                   'Shape Representation',
                   'Shape & Scene Completion',
                   'Shape Reconstruction',
                   '3D Scene Understanding',
                   '3D Scene Reconstruction',
                   'NeRF',
                   'About Human Body',
                   'General Methods',
                   'Others (inc. Networks in Classification, Matching, Registration, Alignment, Depth, Normal, Pose, Keypoints, etc.)',
                   'Survey, Resources and Tools']


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

"""
    File to load dataset based on user control from main file
"""

from data.STOIC.STOIC_Dataset import STOIC_Dataset


def LoadData(DATASET_NAME):
    if DATASET_NAME == "STOIC":
        return STOIC_Dataset(name='STOIC')
    if DATASET_NAME == "PreGraph":
        return STOIC_Dataset(name='PreGraph')





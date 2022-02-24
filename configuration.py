import os
from math import ceil

SEED = 47

DATA_FOLDERS = [
    os.sep.join(["datasets", folder])
                for folder
                in ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace"]
]
DIRECTION_FOLDERS = ["0-back", "1-left", "2-front", "3-right"]

DATASET_SIZES = [0, 0, 0, 408]
# DATASET_SIZES = [912, 216, 294, 408]
DATASET_SIZE = sum(DATASET_SIZES)
TRAIN_PERCENTAGE = 0.85
TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
TRAIN_SIZE = sum(TRAIN_SIZES)
TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i] for i, n in enumerate(DATASET_SIZES)]
TEST_SIZE = sum(TEST_SIZES)

BUFFER_SIZE = DATASET_SIZE
BATCH_SIZE = 1

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

TEMP_FOLDER = "temp-side2side"
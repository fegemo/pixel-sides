import os
from math import ceil

SEED = 47

DATA_FOLDERS = [
    os.sep.join(["datasets", folder])
                for folder
                in ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace", "miscellaneous"]
]

DIRECTION_FOLDERS = ["0-back", "1-left", "2-front", "3-right"]
DATASET_MASK = [0, 0, 1, 0, 0]
# DATASET_MASK = [1, 1, 1, 1, 0]
DATASET_SIZES = [912, 216, 294, 408, 12372]
DATASET_SIZES = [n*m for n, m in zip(DATASET_SIZES, DATASET_MASK)]

DATASET_SIZE = sum(DATASET_SIZES)
TRAIN_PERCENTAGE = 0.85
TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
TRAIN_SIZE = sum(TRAIN_SIZES)
TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i] for i, n in enumerate(DATASET_SIZES)]
# TEST_SIZES = [0, 0, 44, 0, 0]
TEST_SIZE = sum(TEST_SIZES)

BUFFER_SIZE = DATASET_SIZE
BATCH_SIZE = 4

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

# for stargan-based
NUMBER_OF_DOMAINS = len(DIRECTION_FOLDERS)

# for indexed colors
MAX_PALETTE_SIZE = 256
INVALID_INDEX_COLOR = [255, 0, 220, 255]    # some pink

TEMP_FOLDER = "temp-side2side"


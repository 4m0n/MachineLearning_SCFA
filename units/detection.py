import numpy as np
from scipy.special import erf
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from keras import backend as K

class Colors:
    @staticmethod
    def red():
        return (239, 8, 8)

    @staticmethod
    def darkred():
        return (148, 20, 33)

    @staticmethod
    def orange():
        return (255, 134, 57)

    @staticmethod
    def darkorange():
        return (181, 101, 24)

    @staticmethod
    def darkgold():
        return (165, 150, 0)

    @staticmethod
    def yellow():
        return (255, 251, 0)

    @staticmethod
    def lime():
        return (156, 219, 0)

    @staticmethod
    def green():
        return (66, 190, 66)

    @staticmethod
    def darkgreen():
        return (41, 138, 82)

    @staticmethod
    def teal():
        return (41, 77, 74)

    @staticmethod
    def darkteal():
        return (41, 77, 74) #idk

    @staticmethod
    def blue():
        return (66, 109, 239)

    @staticmethod
    def darkblue():
        return (41, 40, 231)

    @staticmethod
    def purple():
        return (90, 0, 165)

    @staticmethod
    def violet():
        return (148, 97, 255)

    @staticmethod
    def turkis():
        return (99, 255, 206)

    @staticmethod
    def white():
        return (255, 255, 255)

    @staticmethod
    def gray():
        return (99, 109, 123)

    @staticmethod
    def magenta():
        return (255, 138, 255)

    @staticmethod
    def pink():
        return (255, 48, 255)
class Colors2:
    @staticmethod
    def red():
        return (240, 41, 40)

    @staticmethod
    def darkred():
        return (150, 48, 58)

    @staticmethod
    def orange():
        return (255, 135, 58)

    @staticmethod
    def darkorange():
        return (182, 109, 51)

    @staticmethod
    def darkgold():
        return (165, 152, 45)

    @staticmethod
    def yellow():
        return (255, 252, 48)

    @staticmethod
    def lime():
        return (157, 220, 43)

    @staticmethod
    def green():
        return (67, 191, 67)

    @staticmethod
    def darkgreen():
        return (88, 139, 86)

    @staticmethod
    def teal():
        return (64, 91, 88) #idk

    @staticmethod
    def blue():
        return (67, 110, 240)

    @staticmethod
    def darkblue():
        return (88, 80, 232)

    @staticmethod
    def purple():
        return (105, 74, 166)

    @staticmethod
    def violet():
        return (149, 98, 255)

    @staticmethod
    def turkis():
        return (111, 255, 207)

    @staticmethod
    def white():
        return (255, 255, 255)

    @staticmethod
    def gray():
        return (99, 109, 123)

    @staticmethod
    def magenta():
        return (255, 138, 255)

    @staticmethod
    def pink():
        return (255, 48, 255)
    @staticmethod
    def eye():
        return (240, 236, 240)

def overlay_image_alpha(img, img_overlay, x, y):
    """Overlay `img_overlay` onto `img` at (x, y) using the alpha channel."""
    if img_overlay.shape[2] == 4:
        alpha_mask_overlay = img_overlay[:, :, 3] / 255.0
        img_overlay_bgr = img_overlay[:, :, :3]
    else:
        alpha_mask_overlay = np.ones(img_overlay.shape[:2], dtype=np.float32)
        img_overlay_bgr = img_overlay

    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay_bgr.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay_bgr.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay_bgr.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay_bgr.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop_bgr = img[y1:y2, x1:x2, :3]
    img_overlay_crop = img_overlay_bgr[y1o:y2o, x1o:x2o]
    alpha_overlay = alpha_mask_overlay[y1o:y2o, x1o:x2o, np.newaxis]

    # Check if the base image has an alpha channel
    has_alpha = img.shape[2] == 4
    if has_alpha:
        alpha_mask_base = img[y1:y2, x1:x2, 3] / 255.0
        # For the resulting alpha channel, take the maximum (less transparent)
        alpha_combined_for_transparency = np.maximum(alpha_mask_overlay[y1o:y2o, x1o:x2o], alpha_mask_base)
        # For blending the BGR channels, use the overlay's alpha channel
        alpha_combined_for_blending = alpha_overlay
    else:
        alpha_combined_for_transparency = alpha_overlay
        alpha_combined_for_blending = alpha_overlay

    alpha_inv = 1.0 - alpha_combined_for_blending
    img_crop_bgr[:] = (alpha_combined_for_blending * img_overlay_crop + alpha_inv * img_crop_bgr).astype(np.uint8)

    # Update the alpha channel if the base image has one
    if has_alpha:
        img[y1:y2, x1:x2, 3] = (alpha_combined_for_transparency * 255).astype(np.uint8)


def apply_tier(tier, unit):

    unit_height, unit_width = unit.shape[0], unit.shape[1]
    tier_height, tier_width = tier.shape[0], tier.shape[1]

    width_start = ((unit_width - 1) // 2) - ((tier_width - 1) // 2)
    width_end = ((unit_width + 1) // 2) + ((tier_width - 1) // 2)

    height_start = unit_height - tier_height + 3

    resized = np.zeros((unit_height + 3, unit_width, 4))
    resized[height_start:, width_start:width_end, :] = tier

    overlay_image_alpha(resized, unit, 0, 0)

    return resized

def amount(unit_time, state, max_amount):
    amount = 0
    if unit_time is 'early':
        amount = np.exp(-(10(state-0.25)**2))
    if unit_time is 'mid':
        amount = np.exp(-(10(state-0.6)**2))
    if unit_time is 'late':
        amount = np.exp(-(10(state-1)**2))
    if unit_time is 'const':
        amount = np.log(69* state+1)/np.log(70)
    return amount * max_amount

def tier(state):
    temp = np.random.random()
    if temp < erf(5*(state - 0.5)):
        return 3
    elif temp < erf(2*(state - 0.1)):
        return 2
    else:
        return 1

def place_base(image, icon, count, width, height, base_center):
    for _ in range(count):
        x_pos, y_pos = np.random.multivariate_normal(
            base_center,
            [[500, 0], [0, 500]]
        )
        x_pos, y_pos = int(x_pos), int(y_pos)
        overlay_image_alpha(image, icon, x_pos, y_pos)

def place_clustered(image, icon, count, width, height):
    cluster_center = None
    for _ in range(count):
        if cluster_center is None or np.random.random() > 0.9:
            cluster_center = [np.random.randint(0, width), np.random.randint(0, height)]
        x_pos, y_pos = np.random.multivariate_normal(
            cluster_center,
            [[500, 0], [0, 500]]
        )
        x_pos, y_pos = int(x_pos), int(y_pos)
        overlay_image_alpha(image, icon, x_pos, y_pos)

def place_evenly(image, icon, count, width, height):
    for _ in range(count):
        x_pos, y_pos = np.random.randint(0, width), np.random.randint(0, height)
        overlay_image_alpha(image, icon, x_pos, y_pos)

def place_units(image, unit_dict, width, height):
    unit_counts = {}
    base_center = [np.random.beta(0.5, 0.5)*width , np.random.beta(0.5, 0.5)*height]  # Common base center

    for unit_name, unit_info in unit_dict.items():
        count = np.random.randint(unit_info['min_count'], unit_info['max_count'] + 1)
        unit_counts[unit_name] = count

        base_count = int(count * unit_info['fraction_base'])
        clustered_count = int(count * unit_info['fraction_clustered'])
        even_count = count - base_count - clustered_count

        # Place units around the base
        if base_count > 0:
            place_base(image, unit_info['icon'], base_count, width, height, base_center)

        # Place units in clusters
        if clustered_count > 0:
            place_clustered(image, unit_info['icon'], clustered_count, width, height)

        # Place units evenly
        if even_count > 0:
            place_evenly(image, unit_info['icon'], even_count, width, height)

    return unit_counts

def data_generator(batch_size, height, width, unit_dict):
    while True:
        X_batch = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_batch = np.zeros((batch_size, len(unit_dict)), dtype=np.int32)
        for i in range(batch_size):
            game = np.zeros((height, width, 3), dtype=np.uint8)
            unit_counts = place_units(game, unit_dict, width, height)

            # Store the counts in y_batch
            for j, (unit_name, count) in enumerate(unit_counts.items()):
                y_batch[i, j] = count

            # Convert to grayscale and normalize
            gray = cv2.cvtColor(game, cv2.COLOR_BGR2GRAY)
            X_batch[i, :, :, 0] = gray.astype(np.float32) / 255.0

        # Prepare y_batch as a tuple of arrays
        y_batch_tuple = tuple(y_batch[:, i].reshape(-1, 1) for i in range(y_batch.shape[1]))

        yield X_batch, y_batch_tuple

units = {
    'engineer': {
        'name': 'engineer',
        'icon': cv2.imread('units/icons/land/engineer.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 50,
        'fraction_base': 0.2,
        'fraction_clustered': 0.3,
        'time': 'const',
        'type' : 'land'
    },
    'mass': {
        'name': 'mass',
        'icon': cv2.imread('units/icons/building/mass.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 30,
        'fraction_base': 0.2,
        'fraction_clustered': 0.0,
        'time': 'const',
        'type' : 'building'
    },
    'tank': {
        'name': 'tank',
        'icon': cv2.imread('units/icons/land/tank.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 100,
        'fraction_base': 0.1,
        'fraction_clustered': 0.5,
        'time': 'const',
        'type' : 'land'
    },
    'air-scout': {
        'name': 'air-scout',
        'icon': cv2.imread('units/icons/air/air-scout.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 30,
        'fraction_base': 0.0,
        'fraction_clustered': 0.0,
        'time': 'const',
        'type' : 'air'
    },
    'interceptor': {
        'name': 'interceptor',
        'icon': cv2.imread('units/icons/air/interceptor.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 30,
        'fraction_base': 0.2,
        'fraction_clustered': 0.3,
        'time': 'early',
        'type' : 'air'
    },
    'torpedo-bomber': {
        'name': 'torpedo-bomber',
        'icon': cv2.imread('units/icons/air/torpedo-bomber.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 20,
        'fraction_base': 0.1,
        'fraction_clustered': 0.3,
        'time': 'late',
        'type' : 'air'
    },
    'land-factory': {
        'name': 'land-factory',
        'icon': cv2.imread('units/icons/building/land-factory.png', cv2.IMREAD_UNCHANGED),
        'min_count': 1,
        'max_count': 10,
        'fraction_base': 1.0,
        'fraction_clustered': 0.0,
        'time': 'const',
        'type' : 'building'
    },
    'air-factory': {
        'name': 'air-factory',
        'icon': cv2.imread('units/icons/building/air-factory.png', cv2.IMREAD_UNCHANGED),
        'min_count': 1,
        'max_count': 3,
        'fraction_base': 1.0,
        'fraction_clustered': 0.0,
        'time': 'const',
        'type' : 'building'
    },
    'assault-bot': {
        'name': 'assault-bot',
        'icon': cv2.imread('units/icons/land/assault-bot.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 30,
        'fraction_base': 0.1,
        'fraction_clustered': 0.4,
        'time': 'early',
        'type' : 'land'
    },
    'aircraft-carrier': {
        'name': 'aircraft-carrier',
        'icon': cv2.imread('units/icons/naval/aircraft-carrier.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 20,
        'fraction_base': 0.1,
        'fraction_clustered': 0.5,
        'time': 'late',
        'type' : 'naval'
    },
    'missile-submarine': {
        'name': 'missile-submarine',
        'icon': cv2.imread('units/icons/naval/missile-submarine.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 40,
        'fraction_base': 0.1,
        'fraction_clustered': 0.4,
        'time': 'late',
        'type' : 'naval'
    },
    'cruiser': {
        'name': 'cruiser',
        'icon': cv2.imread('units/icons/naval/cruiser.png', cv2.IMREAD_UNCHANGED),
        'min_count': 0,
        'max_count': 40,
        'fraction_base': 0.1,
        'fraction_clustered': 0.4,
        'time': 'mid',
        'type' : 'naval'
    }
}

def create_multi_output_model(input_shape, units):
    unit_names = list(units.keys())

    
    inputs = Input(shape=input_shape)

    # Shared convolutional backbone
    x = Conv2D(32, (5, 5), activation='relu')(inputs)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Separate output heads for each unit type
    outputs = []
    for unit_name in unit_names:
        unit_output = Dense(64, activation='relu')(x)
        unit_output = Dropout(0.2)(unit_output)
        unit_count = Dense(1, activation='linear', name=f'{unit_name}_count')(unit_output)
        outputs.append(unit_count)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# define parameters
height = 1080
width = 1920

batch_size = 10
epochs = 10
steps_per_epoch = 20

unit_names = list(units.keys())
generator = data_generator(batch_size, height, width, units)


# Build the model
model = create_multi_output_model((height, width, 1), units)
model.summary()

# Compile with separate losses for each output
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={f'{unit_name}_count': 'mse' for unit_name in unit_names},
    loss_weights={f'{unit_name}_count': 1.0 for unit_name in unit_names},
    metrics={f'{unit_name}_count': ['mae'] for unit_name in unit_names}
)

history = model.fit(
    generator,
    steps_per_epoch = steps_per_epoch,  # Specify the number of batches per epoch
    epochs = epochs,  # Specify the number of epochs
    verbose =1
)

model.save('units/icon_counter.h5')
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
    if unit_time == 'early':
        amount = np.exp(-(10*(state-0.25)**2))
    if unit_time == 'mid':
        amount = np.exp(-(10*(state-0.6)**2))
    if unit_time == 'late':
        amount = np.exp(-(10*(state-1)**2))
    if unit_time == 'const':
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
    # Initialize counts for each type and tier combination
    type_tier_counts = {
        'air': {1: 0, 2: 0, 3: 0},
        'land': {1: 0, 2: 0, 3: 0},
        'naval': {1: 0, 2: 0, 3: 0},
        'building': {1: 0, 2: 0, 3: 0}
    }
    base_center = [np.random.beta(0.5, 0.5)*width , np.random.beta(0.5, 0.5)*height]  # Common base center

    for unit_name, unit_info in unit_dict.items():
        state = np.random.random()  # Generate a random state value
        max_amount = unit_info['max_count']
        count = int(amount(unit_info['time'], state, max_amount))
        tier_value = tier(state)
        tier_icon = tier_icons[tier_value]
        tiered_icon = apply_tier(tier_icon, unit_info['icon'])
        # Update the type_tier_counts dictionary
        unit_type = unit_info['type']
        type_tier_counts[unit_type][tier_value] += count
        base_count = int(count * unit_info['fraction_base'])
        clustered_count = int(count * unit_info['fraction_clustered'])
        even_count = count - base_count - clustered_count
        # Place units around the base
        if base_count > 0:
            place_base(image, tiered_icon, base_count, width, height, base_center)
        # Place units in clusters
        if clustered_count > 0:
            place_clustered(image, tiered_icon, clustered_count, width, height)
        # Place units evenly
        if even_count > 0:
            place_evenly(image, tiered_icon, even_count, width, height)
    # Flatten the type_tier_counts dictionary to a list of counts
    counts = [
        type_tier_counts['air'][1],
        type_tier_counts['air'][2],
        type_tier_counts['air'][3],
        type_tier_counts['land'][1],
        type_tier_counts['land'][2],
        type_tier_counts['land'][3],
        type_tier_counts['naval'][1],
        type_tier_counts['naval'][2],
        type_tier_counts['naval'][3],
        type_tier_counts['building'][1],
        type_tier_counts['building'][2],
        type_tier_counts['building'][3]
    ]
    return counts

def data_generator(batch_size, height, width, unit_dict):
    while True:
        X_batch = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_batch = np.zeros((batch_size, 12), dtype=np.int32)  # 12 counts for type and tier combinations
        for i in range(batch_size):
            game = np.zeros((height, width, 3), dtype=np.uint8)
            unit_counts = place_units(game, unit_dict, width, height)
            # Store the counts in y_batch
            for j, count in enumerate(unit_counts):
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

tier_icons = {
    1 : cv2.imread(f'units/icons/tier-1.png', cv2.IMREAD_UNCHANGED),
    2 : cv2.imread(f'units/icons/tier-2.png', cv2.IMREAD_UNCHANGED),
    3 : cv2.imread(f'units/icons/tier-3.png', cv2.IMREAD_UNCHANGED),
}

def create_multi_output_model(input_shape, units):
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
    # Separate output heads for each type and tier combination
    outputs = []
    output_names = [
        'air_tier1', 'air_tier2', 'air_tier3',
        'land_tier1', 'land_tier2', 'land_tier3',
        'naval_tier1', 'naval_tier2', 'naval_tier3',
        'building_tier1', 'building_tier2', 'building_tier3'
    ]
    for output_name in output_names:
        unit_output = Dense(64, activation='relu')(x)
        unit_output = Dropout(0.2)(unit_output)
        unit_count = Dense(1, activation='linear', name=f'{output_name}_count')(unit_output)
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
output_names = [
    'air_tier1', 'air_tier2', 'air_tier3',
    'land_tier1', 'land_tier2', 'land_tier3',
    'naval_tier1', 'naval_tier2', 'naval_tier3',
    'building_tier1', 'building_tier2', 'building_tier3'
]
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={f'{output_name}_count': 'mse' for output_name in output_names},
    loss_weights={f'{output_name}_count': 1.0 for output_name in output_names},
    metrics={f'{output_name}_count': ['mae'] for output_name in output_names}
)

history = model.fit(
    generator,
    steps_per_epoch = steps_per_epoch,  # Specify the number of batches per epoch
    epochs = epochs,  # Specify the number of epochs
    verbose =1
)

model.save('units/icon_counter.h5')

# # Create the directory if it doesn't exist
# os.makedirs('units/example/', exist_ok=True)

# # Generate and save 20 example images
# for batch in range(2):  # 2 batches of 10 images each
#     X_batch, y_batch = next(generator)
#     for i in range(batch_size):
#         image = X_batch[i, :, :, 0] * 255.0  # Scale back to 0-255
#         image = image.astype(np.uint8)
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels
#         cv2.imwrite(f'units/example/example_{batch * batch_size + i}.png', image)

# def save_tiered_units(unit_dict, output_dir, num_examples=3):
#     # Create the directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Generate and save tiered unit icons
#     for unit_name, unit_info in unit_dict.items():
#         for i in range(num_examples):
#             state = np.random.random()  # Generate a random state value
#             tier_value = tier(state)
#             tier_icon = tier_icons[tier_value]
#             tiered_icon = apply_tier(tier_icon, unit_info['icon'])
#             # Save the tiered icon
#             filename = f"{unit_name}_tier{tier_value}_{i}.png"
#             cv2.imwrite(os.path.join(output_dir, filename), tiered_icon)

# # Assuming units is already defined
# save_tiered_units(units, 'units/example/')

import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
MODEL_PATH = "diabetic_retinopathy_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# Get EfficientNetB0
base_model = model.get_layer("efficientnetb0")

# Find the last convolutional layer
last_conv_layer = None

for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError("Convolutional layer not found")

print("Grad-CAM layer:", last_conv_layer.name)

# Create feature extraction model
grad_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


def make_gradcam_heatmap(img_array, predicted_class):

    with tf.GradientTape() as tape:

        conv_outputs, features = grad_model(
            img_array, training=False
        )

        x = model.get_layer(
            "global_average_pooling2d"
        )(features)

        x = model.get_layer("dropout")(
            x, training=False
        )

        predictions = model.get_layer("dense")(x)

        class_score = predictions[:, predicted_class]

    grads = tape.gradient(class_score, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads, axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def create_gradcam_overlay(image, heatmap):

    heatmap = cv2.resize(
        heatmap, (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap, cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image.astype("uint8"),
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay

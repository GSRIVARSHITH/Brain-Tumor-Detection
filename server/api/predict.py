import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from io import BytesIO
import matplotlib.cm as cm


class BrainTumorPredictor:
    """
    Brain Tumor Classification + Grad-CAM Explainability
    """

    # -----------------------------
    # INIT
    # -----------------------------
    def __init__(
        self,
        model_path: str,
        classes: list,
        img_size=(512, 512),
        confidence_threshold=0.70,
        last_conv_layer_name="conv2d_1"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError("❌ Model file not found")

        self.model = tf.keras.models.load_model(model_path)
        self.classes = classes
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.last_conv_layer_name = last_conv_layer_name

        self._init_gradcam()

    # -----------------------------
    # GRAD-CAM INITIALIZATION
    # -----------------------------
    def _init_gradcam(self):
        layer_names = [layer.name for layer in self.model.layers]

        if self.last_conv_layer_name not in layer_names:
            print(f"⚠️ Grad-CAM disabled: {self.last_conv_layer_name} not found")
            self.gradcam_ready = False
            return

        conv_index = layer_names.index(self.last_conv_layer_name)

        self.conv_model = tf.keras.models.Model(
            self.model.inputs,
            self.model.get_layer(self.last_conv_layer_name).output
        )

        self.classifier_layers = self.model.layers[conv_index + 1 :]
        self.gradcam_ready = True

    # -----------------------------
    # IMAGE LOADING
    # -----------------------------
    def load_and_preprocess_image(self, img_path: str):
        if not os.path.exists(img_path):
            raise FileNotFoundError("❌ Image path not found")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("❌ Unable to read image")

        img = cv2.resize(img, self.img_size)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)

    # -----------------------------
    # MRI VALIDATION
    # -----------------------------
    def is_valid_mri(self, img):
        try:
            gray = cv2.cvtColor(
                (img[0] * 255).astype(np.uint8),
                cv2.COLOR_BGR2GRAY
            )

            mean, std = np.mean(gray), np.std(gray)

            return not (mean < 30 or mean > 220 or std < 10)
        except Exception:
            return False

    # -----------------------------
    # GRAD-CAM HEATMAP
    # -----------------------------
    def generate_gradcam_heatmap(self, img, class_index):
        if not self.gradcam_ready:
            return None

        img_tensor = tf.convert_to_tensor(img)

        with tf.GradientTape() as tape:
            conv_output = self.conv_model(img_tensor)
            tape.watch(conv_output)

            x = conv_output
            for layer in self.classifier_layers:
                x = layer(x)

            class_score = x[:, class_index]

        grads = tape.gradient(class_score, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8

        return heatmap.numpy()

    # -----------------------------
    # HEATMAP OVERLAY
    # -----------------------------
    def overlay_gradcam(self, original_img, heatmap, alpha=0.4):
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = np.uint8(255 * jet_heatmap)

        overlay = cv2.addWeighted(original_img, 1 - alpha, jet_heatmap, alpha, 0)
        return overlay

    # -----------------------------
    # PREDICTION
    # -----------------------------
    def predict(self, img):
        if not self.is_valid_mri(img):
            return self._error_response("Invalid brain MRI image")

        preds = self.model.predict(img, verbose=0)[0]
        class_index = int(np.argmax(preds))
        confidence = float(preds[class_index])
        label = self.classes[class_index]

        gradcam_b64, gradcam_error = None, None

        try:
            original_img = (img[0] * 255).astype("uint8")
            heatmap = self.generate_gradcam_heatmap(img, class_index)

            if heatmap is not None:
                gradcam_img = self.overlay_gradcam(original_img, heatmap)
                gradcam_b64 = self._encode_image(gradcam_img)
        except Exception as e:
            gradcam_error = str(e)

        if confidence < self.confidence_threshold:
            return self._response(
                status="low_confidence",
                label=label,
                confidence=confidence,
                message="Low confidence prediction. Consult a radiologist.",
                gradcam=gradcam_b64,
                error=gradcam_error
            )

        return self._response(
            status="success",
            label=label,
            confidence=confidence,
            message="Prediction successful",
            gradcam=gradcam_b64,
            error=gradcam_error
        )

    # -----------------------------
    # HELPERS
    # -----------------------------
    def _encode_image(self, img):
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")

    def _response(self, status, label, confidence, message, gradcam, error):
        return {
            "status": status,
            "prediction": label,
            "confidence": round(confidence, 2),
            "message": message,
            "gradcam_image": gradcam,
            "gradcam_error": error
        }

    def _error_response(self, msg):
        return {
            "status": "error",
            "prediction": None,
            "confidence": 0,
            "message": msg,
            "gradcam_image": None
        }


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    CLASSES = ["Glioma", "Meningioma", "No_Tumor", "Pituitary"]

    predictor = BrainTumorPredictor(
        model_path="model/cnn_model.h5",
        classes=CLASSES
    )

    img_path = input("Enter MRI image path: ").strip()
    img = predictor.load_and_preprocess_image(img_path)

    result = predictor.predict(img)

    print("\n Prediction Result")
    for k, v in result.items():
        print(f"{k}: {v}")

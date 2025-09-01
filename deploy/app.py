import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Config

DEVICE = torch.device("cpu")
IMG_SIZE = 300
MODEL_PATH = "models/efficientnet_b3_cbam_mixup_cutmix.pt"

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]



# CBAM Attention

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, 1, keepdim=True)
        max_out, _ = torch.max(x, 1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], 1)))

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# Model

class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b3(weights=None)
        in_features = backbone.classifier[1].in_features
        self.backbone = backbone
        self.cbam = CBAM(in_features)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


# Load model

model = EfficientNet_CBAM(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# Transform

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Disease Information 

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "symptoms": "Small brown scabby spots on fruit; Leaf drop exposing fruit to sunscald",
        "prevention": "Use certified seed/transplants; Sanitize tools; Rotate crops; Avoid wet handling"
    },
    "Pepper__bell___healthy": {
        "symptoms": "Healthy leaf with no symptoms",
        "prevention": "Maintain good growing conditions"
    },
    "Potato___Early_blight": {
        "symptoms": "Dark concentric-ring lesions on leaves, stems, fruit; Defoliation",
        "prevention": "Rotate crops; Remove debris; Use resistant varieties; Apply protectant fungicides"
    },
    "Potato___Late_blight": {
        "symptoms": "Water-soaked lesions quickly turning brown with possible white mold",
        "prevention": "Use disease-free seed; Sterilize soil; Remove infected foliage; Apply fungicides"
    },
    "Potato___healthy": {
        "symptoms": "Healthy leaf with no symptoms",
        "prevention": "Maintain good growing conditions"
    },
    "Tomato_Bacterial_spot": {
        "symptoms": "Dark circular spots with yellow halo on leaves; Scabby fruit; Defoliation",
        "prevention": "Use certified seed; Sanitize tools; Rotate crops; Avoid wet handling"
    },
    "Tomato_Early_blight": {
        "symptoms": "Bull’s-eye lesions on older leaves; Defoliation and sunscald",
        "prevention": "Remove debris; Improve airflow; Rotate crops; Apply fungicide early"
    },
    "Tomato_Late_blight": {
        "symptoms": "Rapid brown decay with water-soaked spots; Possible white mold",
        "prevention": "Use resistant varieties; Avoid overhead watering; Prune; Apply fungicide"
    },
    "Tomato_Leaf_Mold": {
        "symptoms": "Yellow patches with velvety mold underside; Defoliation",
        "prevention": "Improve ventilation; Avoid overhead watering; Remove affected leaves"
    },
    "Tomato_Septoria_leaf_spot": {
        "symptoms": "Circular leaf spots with dark centers and pale margins; Defoliation",
        "prevention": "Remove infected debris; Improve airflow; Use fungicide if needed"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "symptoms": "Yellow stippling on leaves; Webbing; Leaf browning",
        "prevention": "Use miticides or horticultural oils; Introduce predators; Reduce leaf dryness"
    },
    "Tomato__Target_Spot": {
        "symptoms": "Circular lesions with concentric rings on leaves/fruit",
        "prevention": "Remove debris; Apply fungicide; Rotate crops; Ensure airflow"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "symptoms": "Leaves curl, yellow; Stunted growth; Flower drop",
        "prevention": "Control whiteflies; Use resistant cultivars; Remove weeds"
    },
    "Tomato__Tomato_mosaic_virus": {
        "symptoms": "Mosaic-pattern leaves; Distortion; Stunting",
        "prevention": "Use clean seed; Sanitize tools; Avoid tobacco contact"
    },
    "Tomato_healthy": {
        "symptoms": "Healthy leaf with no symptoms",
        "prevention": "Maintain optimal growing conditions"
    }
}

# Flask API

app = Flask(__name__)

def predict_image(image):
    img = Image.open(image).convert("RGB")
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

        confs, indices = torch.topk(probs, 3)

    results = []
    for i in range(3):
        cls = CLASS_NAMES[indices[0][i].item()]
        entry = {
            "class": cls,
            "probability": float(confs[0][i].item()),
            "info": DISEASE_INFO.get(cls, {})
        }
        results.append(entry)

    return results

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    results = predict_image(file)
    return jsonify({"predictions": results})

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

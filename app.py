import os
import numpy as np
import torch
import h5py
from datetime import datetime
from PIL import Image
from flask import Flask, send_from_directory, request, jsonify
from torch import nn
from flask_cors import CORS

# Production configuration
class Config:
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', './Test_Chunks')
    INPUT_DIR = os.environ.get('INPUT_DIR', './INPUT')
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './static/output')

# UNet model definition
class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(64, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def center_crop(self, enc_feat, target_shape):
        _, _, h, w = enc_feat.shape
        target_h, target_w = target_shape
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return enc_feat[:, :, start_h:start_h+target_h, start_w:start_w+target_w]

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        enc2_cropped = self.center_crop(enc2, dec2.shape[2:])
        dec2 = self.decoder2(torch.cat((dec2, enc2_cropped), dim=1))
        dec1 = self.upconv1(dec2)
        enc1_cropped = self.center_crop(enc1, dec1.shape[2:])
        dec1 = self.decoder1(torch.cat((dec1, enc1_cropped), dim=1))
        return self.final_conv(dec1)

# Environment setup
app = Flask(__name__)
app.config.from_object(Config)

# Configure CORS - use Flask-CORS only, no manual headers
CORS(app, 
     origins=[
         "https://teamlava-bah-25-ps5.vercel.app",
         "https://*.vercel.app",
         "https://*.onrender.com",
         "http://localhost:5173",
         "http://localhost:5174",
         "http://localhost:3000",
         "http://127.0.0.1:5173",
         "http://127.0.0.1:5174",
         "http://127.0.0.1:3000"
     ],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

# Paths and model loading
checkpoint_dir = Config.CHECKPOINT_DIR
input_dir = Config.INPUT_DIR
output_dir = Config.OUTPUT_DIR

# Create directories with error handling
try:
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Created directories: {checkpoint_dir}, {input_dir}, {output_dir}")
except Exception as e:
    print(f"⚠️ Error creating directories: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load models with better error handling
models = []
model_paths = []

if os.path.exists(checkpoint_dir):
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if model_files:
        model_paths = sorted([os.path.join(checkpoint_dir, f) for f in model_files])
        for path in model_paths:
            try:
                model = UNet().to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                models.append(model)
                print(f"✅ Loaded model: {os.path.basename(path)}")
            except Exception as e:
                print(f"⚠️ Error loading model from {path}: {e}")
    else:
        print(f"⚠️ No .pth files found in {checkpoint_dir}")
else:
    print(f"⚠️ Checkpoint directory does not exist: {checkpoint_dir}")

print(f"✅ Total models loaded: {len(models)}")

input_paths = []
pred_path = ""

@app.route("/", methods=["GET"])
def index():
    """Main endpoint with health and API information"""
    try:
        # Check if models are loaded
        models_loaded = len(models) > 0
        
        # Check if directories exist
        dirs_exist = (
            os.path.exists(checkpoint_dir) and 
            os.path.exists(input_dir) and 
            os.path.exists(output_dir)
        )
        
        return jsonify({
            "message": "Satellite Change Detection API",
            "status": "running",
            "version": "1.0.0",
            "endpoints": ["/", "/upload", "/static/output/<filename>", "/test"],
            "health": {
                "status": "healthy" if dirs_exist else "unhealthy",
                "timestamp": str(datetime.now()),
                "models_loaded": models_loaded,
                "models_count": len(models),
                "directories_exist": dirs_exist,
                "prediction_ready": models_loaded,
                "device": str(device),
                "note": "Models required for prediction functionality. Server is running but predictions disabled." if not models_loaded else "All systems operational"
            }
        })
        
    except Exception as e:
        return jsonify({
            "message": "Satellite Change Detection API",
            "status": "error",
            "error": str(e),
            "timestamp": str(datetime.now())
        }), 500

@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "message": "Server is working correctly!",
        "timestamp": str(datetime.now()),
        "models_loaded": len(models) > 0,
        "models_count": len(models)
    })

@app.route("/upload", methods=["POST"])
def upload():
    global input_paths, pred_path

    for f in os.listdir(input_dir):
        if f.endswith(".h5"):
            try:
                os.remove(os.path.join(input_dir, f))
            except Exception: pass

    for f in os.listdir(output_dir):
        try:
            os.remove(os.path.join(output_dir, f))
        except Exception: pass

    files = request.files.getlist("files")
    for f in files:
        f.save(os.path.join(input_dir, f.filename))

    if not models:
        return jsonify({"error": "No models loaded."}), 503

    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.h5')])
    if not input_files:
        return jsonify({"error": "No .h5 files found."}), 400

    input_stack = []
    for fpath in input_files:
        with h5py.File(fpath, 'r') as f:
            vis = np.squeeze(f['IMG_VIS'][:])
            swir = np.squeeze(f['IMG_SWIR'][:])
            tir1 = np.squeeze(f['IMG_TIR1'][:])
            tir2 = np.squeeze(f['IMG_TIR2'][:])
            wv = np.squeeze(f['IMG_WV'][:])
            stacked = np.stack([vis, swir, tir1, tir2, wv], axis=0)
            input_stack.append(stacked.astype(np.float32))

    input_stack = np.stack(input_stack)
    input_stack /= np.max(input_stack)
    X = torch.tensor(input_stack).mean(dim=0).unsqueeze(0).to(device)

    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(m(X).cpu())

    ensemble_output = torch.mean(torch.stack(preds), dim=0).squeeze(0)
    input_paths = []
    for i in range(min(5, input_stack.shape[0])):
        vis_band = input_stack[i, 0]
        vis_norm = (255 * (vis_band - vis_band.min()) / np.ptp(vis_band)).astype(np.uint8)
        img = Image.fromarray(vis_norm)
        path = os.path.join(output_dir, f"Input_VIS_Frame_{i+1}.jpg")
        img.save(path)
        input_paths.append(f"static/output/Input_VIS_Frame_{i+1}.jpg")

    pred_np = ensemble_output[0].numpy()
    pred_norm = (255 * (pred_np - pred_np.min()) / np.ptp(pred_np)).astype(np.uint8)
    img = Image.fromarray(pred_norm)
    pred_path = os.path.join(output_dir, "Predicted_VIS_Frame.jpg")
    img.save(pred_path)

    return jsonify({
        "input_images": input_paths,
        "prediction": "static/output/Predicted_VIS_Frame.jpg"
    })

@app.route("/static/output/<filename>")
def get_image(filename):
    return send_from_directory(output_dir, filename)

@app.route("/download/<filename>")
def download_image(filename):
    return send_from_directory(output_dir, filename, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)

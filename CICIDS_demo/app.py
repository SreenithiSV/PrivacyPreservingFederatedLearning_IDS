from flask import Flask, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

app = Flask(__name__)


with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

INPUT_DIM = scaler.mean_.shape[0]
OUTPUT_DIM = len(le.classes_)

feature_names = np.load("feature_names.npy", allow_pickle=True)

class Client0_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, OUTPUT_DIM)
        )
    def forward(self, x):
        return self.net(x)


class Client1_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_DIM)
        )
    def forward(self, x):
        return self.net(x)


class Client2_ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 256)
        self.fc2 = nn.Linear(256, OUTPUT_DIM)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Client3_BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, OUTPUT_DIM)
    def forward(self, x):
        _, h = self.gru(x.unsqueeze(-1))
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))


class Client4_TabAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_embed = nn.Linear(1, 32)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(32)
        self.fc = nn.Sequential(
            nn.Linear(INPUT_DIM * 32, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_DIM)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.ln(x + attn_out)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Client5_CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * INPUT_DIM, 64)
        self.fc2 = nn.Linear(64, OUTPUT_DIM)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

models = torch.load("models.pth", weights_only=False)

demo_samples = np.load("demo_samples.npy")
demo_labels = np.load("demo_labels.npy", allow_pickle=True)
demo_info = np.load("demo_info.npy", allow_pickle=True)

def ensemble_predict(models, X):
    logits_sum = None
    with torch.no_grad():
        for model in models.values():
            model.eval()
            logits = model(X)
            logits_sum = logits if logits_sum is None else logits_sum + logits
    return torch.argmax(logits_sum / len(models), dim=1).numpy()

def predict_single(sample_row):
    sample_scaled = scaler.transform(sample_row.reshape(1, -1))
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    pred = ensemble_predict(models, sample_tensor)
    label = le.inverse_transform(pred)

    return label[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/labels")
def get_labels():
    return jsonify(demo_labels.tolist())

@app.route("/predict/<int:idx>")
def predict(idx):
    sample = demo_samples[idx]
    actual = demo_labels[idx]

    if isinstance(actual, (np.integer, int)):
        actual_label = le.inverse_transform([actual])[0]
    else:
        actual_label = str(actual)

    predicted = predict_single(sample)

    return jsonify({
        "actual": actual_label,
        "predicted": str(predicted),
        "features": sample.tolist(),
        "feature_names": feature_names.tolist(),
        "info": demo_info[idx]
    })
 
if __name__ == "__main__":
    app.run(debug=True)
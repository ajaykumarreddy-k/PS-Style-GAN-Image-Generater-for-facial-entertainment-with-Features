# 🎨 Facial Entertainment GAN (Enhanced)

This repository is an evolution of the [original PS-Style-GAN project](https://github.com/ajaykumarreddy-k/PS-Style-GAN-Image-Generater-for-facial-entertainment-). While the base project focused on high-quality hybrid sketch generation, this version serves as a dedicated environment for implementing advanced facial manipulation features.

---

## 🌟 Feature Roadmap

| Feature | Description | Status |
| --- | --- | --- |
| **Hybrid Sketch** | Optimized GAN-based artistic shading | ✅ |
| **Face Morphing** | Seamlessly transition between facial identities | 🛠️ |
| **Emotion Change** | Modify expressions via latent space manipulation | 🛠️ |
| **Enhanced UI** | Updated Gradio interface with multi-modal support | 🛠️ |

---

## 🚀 Quick Start

### 🖥️ Web Interface

Run the production-ready Gradio app to use the current hybrid sketch generator:

```bash
# Clone the repository
git clone https://github.com/ajaykumarreddy-k/PS-Style-GAN-Image-Generater-for-facial-entertainment-with-Features.git

# Install dependencies
pip install -r requirements.txt

# Launch the app
python gradio_app.py

```

> [!TIP]
> Once running, access the local dashboard at: **http://localhost:7860**

### 🧪 Batch Processing

To generate multiple variations with different AI strengths (30%, 35%, 40%) at once:

```bash
python enhanced_hybrid.py

```

---

## 📦 Repository Structure

```text
.
├── gradio_app.py             # Main web interface
├── enhanced_hybrid.py        # Batch processing script
├── final_model_SHADING.pth   # Core GAN model weights (46 MB)
├── requirements.txt          # Environment dependencies
└── test_face.jpg             # Sample input for testing

```

---

## ⚙️ Model Presets (ENHANCED_35)

| Parameter | Value |
| --- | --- |
| **AI Strength** | 35% (Optimal Balance) |
| **Blur Kernel** | 18 (Sharp Definition) |
| **Preprocessing** | Histogram Eq. + Sharpening |
| **Contrast Ratio** | AI 1.15x / Final 1.05x |

---

## 👥 The Team

* **Project Lead:** [Ajay Kumar Reddy K](https://github.com/ajaykumarreddy-k)
* **Member:** Sameer Raja E
* **Member:** Jeshiba Fedorah
* **Member:** Thanvarshini V R

---

**Status:** In-Development (Feature Addition Phase) 🚀

Would you like me to generate a **custom banner image** for the top of this README to make it even more visually striking?

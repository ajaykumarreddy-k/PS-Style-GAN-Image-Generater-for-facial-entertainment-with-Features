# 🎨 Facial Entertainment GAN (Enhanced)

This repository is an evolution of the [original PS-Style-GAN project](https://github.com/ajaykumarreddy-k/PS-Style-GAN-Image-Generater-for-facial-entertainment-). While the base project focused on high-quality hybrid sketch generation, this version serves as a dedicated environment for implementing advanced facial manipulation features.

## 🌟 New Feature Roadmap

This repo is actively being updated to include:

* [ ] **Face Morphing**: Seamlessly transition between two facial identities.
* [ ] **Emotion Change**: Modify facial expressions (e.g., Happy, Sad, Angry) using GAN latent space manipulation.
* [ ] **Enhanced UI**: Updated Gradio interface to support multi-modal inputs.

## 🚀 Quick Start

### Web Interface

Run the production-ready Gradio app to use the current hybrid sketch generator:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
python gradio_app.py

```

Default URL: **http://localhost:7860**

### Batch Processing

To generate multiple variations with different AI strengths (30%, 35%, 40%) at once:

```bash
python enhanced_hybrid.py

```

## 📦 Repository Structure

| File | Role |
| --- | --- |
| `gradio_app.py` | Main web interface for real-time generation |
| `enhanced_hybrid.py` | Script for batch processing and testing |
| `final_model_SHADING.pth` | Core GAN model (46 MB) |
| `requirements.txt` | Project dependencies |
| `test_face.jpg` | Sample input for testing |

## ⚙️ Model Configuration (Current)

The current implementation uses the **ENHANCED_35** preset for optimal sketch quality:

* **AI Strength**: 35% (Balanced detail)
* **Blur Kernel**: 18 (Sharp line definition)
* **Preprocessing**: Integrated Histogram Equalization & Sharpening
* **Contrast Ratios**: AI 1.15x / Final 1.05x

## 🔧 Development & Extension

Since this is a dedicated repo for new features, feel free to:

1. **Experiment with Latent Directions**: Edit the processing logic in `gradio_app.py` to test emotion vectors.
2. **Swap Models**: Replace `final_model_SHADING.pth` with your own custom-trained weights.
3. **UI Customization**: Use the Gradio layout blocks to add sliders for the new morphing features.

---

**Project Lead:** Ajay Kumar Reddy K
 **Project Member:** Sameer Raja E
**Project Member:** Jeshiba Fedorah 
**Project Member:** Thanvarshini V R

**Status:** In-Development (Feature Addition Phase) 🚀

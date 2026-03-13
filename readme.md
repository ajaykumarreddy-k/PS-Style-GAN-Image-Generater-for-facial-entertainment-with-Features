# 🧬 Project Vibe × PS-Style-GAN
### *Identity-Preserving Face Morphing with Neural Sketch Synthesis*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![dlib](https://img.shields.io/badge/dlib-008000?style=for-the-badge&logo=c%2B%2B&logoColor=white)](http://dlib.net)
[![License](https://img.shields.io/badge/License-Institutional-blue?style=for-the-badge)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)]()
[![Patent](https://img.shields.io/badge/Patent-Pending-yellow?style=for-the-badge)]()

---

## 📌 Table of Contents

- [What Is This?](#what-is-this)
- [The Novelty — What Nobody Saw Coming](#the-novelty)
- [Pipeline Architecture](#pipeline-architecture)
- [File Structure](#file-structure)
- [Component Deep Dive](#component-deep-dive)
  - [Stage 1 — Landmark Detection](#stage-1--landmark-detection)
  - [Stage 2 — Hair & Boundary Anchoring](#stage-2--hair--boundary-anchoring)
  - [Stage 3 — LAB Color Transfer](#stage-3--lab-color-transfer)
  - [Stage 4 — Delaunay Morphing](#stage-4--delaunay-morphing)
  - [Stage 5 — Neural Sketch Synthesis](#stage-5--neural-sketch-synthesis)
- [Model Architecture](#model-architecture)
- [Gradio UI](#gradio-ui)
- [Installation](#installation)
- [Usage](#usage)
- [Known Limitations](#known-limitations)
- [Team](#team)

---

## What Is This?

**Project Vibe × PS-Style-GAN** is a full-stack face morphing and neural sketch synthesis pipeline that takes two face photographs and produces:

1. A **perceptually seamless morphed face** — geometrically warped and color-balanced between the two subjects
2. A **neural sketch / police-style portrait** — rendered from the morph using a custom-trained GAN with VGG skip-connection architecture

This is not a simple alpha-blend. Every stage of the pipeline is architecturally deliberate. The color space is corrected *before* the warp (not after), the triangulation is anchored to prevent hairline collapse, and the sketch GAN receives the morph intermediate — not the raw source photo.

---

## The Novelty

> *"What nobody saw coming"*

There are three things about this pipeline that are genuinely non-obvious and haven't been packaged together before:

### 1. 🔬 Ellipse-Masked LAB Transfer as Morph Preprocessing

Every existing morphing pipeline either:
- Ignores color transfer entirely (produces hue-mismatched seams), or
- Applies color transfer *after* the warp (too late — artifacts are already baked in)

We apply LAB color transfer **before** the Delaunay warp, scoped to a **face-geometry-driven ellipse mask** derived from the 68-point landmark set. This means:

- The color correction is spatially aware — only the face region is corrected, not hair/background
- The Delaunay triangulation receives inputs that are already perceptually close in LAB space
- Seam artifacts at the blend boundary are dramatically reduced

This specific ordering (LAB-masked-transfer → warp) is the core preprocessing innovation.

### 2. 📐 Jaw-Geometry-Anchored Hair Point Generation

Standard Delaunay morphing breaks at the hairline because boundary triangulation near the top of the frame becomes degenerate — long thin triangles that produce visible warping artifacts.

Our `add_hair_and_boundary` function generates **9 synthetic hair anchor points** using a trigonometric arc formula derived from jaw landmark geometry:

```
radius_x = face_width × 0.65
radius_y = face_width × 0.85
arc offset = face_width × 0.15 (upward shift)
```

These points are computed dynamically per-face, not hardcoded, meaning they scale with subject head size. This preserves the hair/forehead region across the morph without any manual tuning.

### 3. 🧠 Morph Intermediate as GAN Input (Not Source Photo)

Every existing sketch-from-photo GAN pipeline takes a **raw photograph** as input. We feed the **morphed face intermediate** — a geometrically and chromatically blended composite — into the sketch GAN.

This means:
- The sketch output encodes identity information from **both subjects simultaneously**
- The GAN's shading adapts to the blended facial geometry, not a single person's structure
- The result is a sketch of a face that *does not exist* — it is genuinely synthetic

This is the pipeline topology that constitutes the primary patent claim.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│   Subject 1 (PIL Image)              Subject 2 (PIL Image)           │
└────────────────┬────────────────────────────┬───────────────────────┘
                 │                            │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1 — SMART CROP                              │
│   ImageOps.fit() → 256×256  (center=0.5, 0.2 face-bias crop)        │
└────────────────┬────────────────────────────┬───────────────────────┘
                 │                            │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 2 — LANDMARK DETECTION (dlib)                     │
│   68-point facial landmark detection                                 │
│   Fallback: 2x upsampling if face not detected at 1x                 │
└────────────────┬────────────────────────────┬───────────────────────┘
                 │                            │
                 ▼                            │
┌────────────────────────────────────────┐   │
│   STAGE 3 — LAB COLOR TRANSFER         │   │
│   Ellipse mask from landmark geometry  │   │
│   Transfer a,b channels from Sub2→Sub1 │   │
│   Strength-weighted face-only blend    │   │
└────────────────┬───────────────────────┘   │
                 │                            │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│         STAGE 4 — HAIR & BOUNDARY ANCHOR GENERATION                  │
│   9 trigonometric hair points (jaw-geometry-derived)                 │
│   16 image boundary points                                           │
│   Total: 68 + 9 + 16 = 93 control points per subject                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 5 — DELAUNAY TRIANGULATION & WARP                 │
│   Midpoint mesh: pm = (1-α)×pts1 + α×pts2                           │
│   Affine warp of each triangle: Sub1→midmesh, Sub2→midmesh          │
│   Alpha blend in warped space: (1-α)×warp1 + α×warp2               │
│   Final ellipse-mask composite with LAB-balanced background          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                          MORPHED FACE (RGB)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 6 — CV2 BASE SKETCH                               │
│   Histogram equalization → Inverted Gaussian blur → Dodge divide     │
│   Sharpening kernel applied                                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│         STAGE 7 — GAN SKETCH SYNTHESIS (SketchGenerator)             │
│   VGG19 Encoder (frozen) → 3 feature pyramid levels                 │
│   SafeFusionModule: stochastic noise injection                       │
│   ConvTranspose decoder with skip connections                        │
│   Alpha blend: GAN output × strength + CV2 × (1-strength)           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                        NEURAL SKETCH OUTPUT
```

---

## File Structure

```
project-vibe-ps-style-gan/
│
├── app.py                          # Main pipeline + Gradio UI (this file)
├── final_model_SHADING.pth         # Trained GAN weights (ENHANCED_35 checkpoint)
├── shape_predictor_68_face_landmarks.dat  # dlib landmark model (auto-downloaded)
│
├── README.md                       # This file
└── requirements.txt                # Dependencies
```

---

## Component Deep Dive

### Stage 1 — Landmark Detection

**Function:** `get_landmarks(img_rgb)`

Uses `dlib`'s HOG-based frontal face detector + 68-point shape predictor.

```python
faces = detector(gray, 1)          # First pass: normal resolution
if len(faces) == 0:
    faces = detector(gray, 2)      # Fallback: 2x upsampled (catches small/tilted faces)
```

Returns a `(68, 2)` numpy array of `(x, y)` coordinates covering:
- Jaw line: points 0–16
- Eyebrows: 17–26
- Nose: 27–35
- Eyes: 36–47
- Mouth: 48–67

If detection fails entirely, the pipeline returns an error message rather than silently producing garbage output.

---

### Stage 2 — Hair & Boundary Anchoring

**Function:** `add_hair_and_boundary(pts, w, h)`

This is one of the key innovations. Takes the 68 landmark points and appends 25 additional control points:

**9 Hair Points** — Generated via trigonometric arc:
```python
for angle in np.linspace(np.pi, 0, 9):
    rx = face_width * 0.65
    ry = face_width * 0.85
    hx = face_center[0] + rx * cos(angle)
    hy = face_center[1] - ry * sin(angle) - (face_width * 0.15)
```

These trace a semi-ellipse above the forehead, anchored to jaw width. They scale automatically with the subject's head size — no hardcoded pixel positions.

**16 Boundary Points** — Fixed corners, midpoints and quarter-points of the image frame. Ensures Delaunay triangulation covers the entire frame without degenerate edge triangles.

**Total control points per subject: 93**

---

### Stage 3 — LAB Color Transfer

**Function:** `lab_transfer(src, ref, strength=0.6)`

Standard LAB color transfer exists in literature. What's novel here is the **spatial scoping**:

```
1. Convert src and ref to LAB color space
2. Detect face landmarks in src
3. Generate ellipse mask from landmark bounding geometry
4. Apply Gaussian blur to mask edges (soft boundary)
5. Transfer a,b channels ONLY inside the mask
6. Blend L channel 70% src / 30% ref (preserves subject's lighting structure)
```

This means:
- Hair color is **not** affected (outside ellipse)
- Background is **not** affected
- Skin tone is harmonized between subjects
- Lighting direction is mostly preserved (70/30 L channel blend)

The `strength` parameter (default 0.4 in morphing context) controls how aggressively the color space is pulled toward the reference subject.

---

### Stage 4 — Delaunay Morphing

**Functions:** `compute_triangles()`, `warp_triangle()`, `process_morph()`

**How Delaunay morphing works:**

1. Compute midpoint mesh: `pm = (1-α) × pts1 + α × pts2`
2. Triangulate the midpoint mesh using `cv2.Subdiv2D`
3. For each triangle in the midpoint mesh:
   - Find the corresponding triangle in Subject 1's point set
   - Find the corresponding triangle in Subject 2's point set
   - Affine-warp Subject 1's triangle region → midpoint triangle → `warp1`
   - Affine-warp Subject 2's triangle region → midpoint triangle → `warp2`
4. Alpha blend: `blend = (1-α) × warp1 + α × warp2`
5. Composite the face region back over the LAB-balanced background using the ellipse mask

**The `warp_triangle` function:**
- Computes bounding rect for each triangle pair
- Uses `cv2.getAffineTransform` for the warp matrix
- Uses `cv2.BORDER_REFLECT_101` to prevent edge tearing
- Applies a filled polygon mask to blend only inside the triangle

**Alpha slider meaning:**
- `α = 0.0` → 100% Subject 1 structure and color
- `α = 0.5` → Perfect 50/50 midpoint morph
- `α = 1.0` → 100% Subject 2 structure and color

---

### Stage 5 — Neural Sketch Synthesis

**Function:** `process_sketch(morphed_cv2, blur_kernel, ai_strength)`

**CV2 base sketch pipeline:**
```
1. RGB → Grayscale
2. Histogram equalization (CLAHE-like normalization)
3. Invert → Gaussian blur the inverted image
4. Dodge blend: gray / (255 - blurred)  [produces pencil line effect]
5. Sharpening kernel convolution
```

**GAN enhancement:**
The CV2 sketch is passed to `SketchGenerator` which adds learned shading and texture. The final output is a weighted blend:

```python
final = cv2.addWeighted(gan_output, ai_strength/100, cv2_sketch, 1 - ai_strength/100, 0)
```

`ai_strength = 35` (default, ENHANCED_35 checkpoint) means 35% GAN shading, 65% sharp CV2 lines. This ratio was tuned to produce police-sketch aesthetics without over-smoothing the line work.

---

## Model Architecture

### `SketchGenerator`

```
Input: RGB image (3 × 256 × 256), normalized to [-1, 1]

ENCODER (VGG19 features, frozen weights):
  enc1: VGG layers 0–3   → (64 × H × W)    [low-level edges]
  enc2: VGG layers 4–13  → (128 × H/2 × W/2) [mid-level structure]
  enc3: VGG layers 14–23 → (256 × H/4 × W/4) [high-level semantics]

FUSION:
  SafeFusionModule(512):
    → adds scaled random noise: x + (noise × learnable_scale)
    → introduces stochastic variation in shading tone

DECODER (skip connections):
  up1: ConvTranspose(512→256, 4×4, stride=2) + InstanceNorm + ReLU
       + concat with enc2 features → (512 × H/2 × W/2)

  up2: ConvTranspose(512→128, 4×4, stride=2) + InstanceNorm + ReLU
       + concat with enc1 features → (192 × H × W)

  up3: ConvTranspose(192→64, 4×4, stride=2) + InstanceNorm + ReLU

  final: Conv2d(64→1, 3×3) + Tanh → (1 × 256 × 256) [grayscale sketch]

Output: Grayscale sketch, denormalized to [0, 255]
```

### `SafeFusionModule`

```python
self.noise_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

def forward(self, x):
    noise = torch.randn(x.size(0), 1, x.size(2), x.size(3))
    return x + (noise * self.noise_scale)
```

The `noise_scale` starts at zero (no effect at initialization) and is learned during training. This means the module can learn *how much* stochastic variation to inject per channel — if the task doesn't benefit from noise, it stays near zero. This is inspired by StyleGAN's noise injection but simplified for a discriminative sketch task.

---

## Gradio UI

The interface is built with `gr.Blocks` and has two columns:

**Left — Controls:**
| Control | Range | Default | Effect |
|---|---|---|---|
| Subject 1 | Image upload | — | Structure/pose donor |
| Subject 2 | Image upload | — | Feature/style donor |
| Morph Balance (α) | 0.0 – 1.0 | 0.5 | Controls identity blend |
| Line Sharpness | 10 – 30 | 18 | CV2 blur kernel size |
| GAN Shading Strength | 0 – 100% | 35% | GAN vs CV2 blend ratio |

**Right — Outputs:**
- Morphed Face (RGB) — the Delaunay blend result
- Neural Sketch Output — the GAN-enhanced sketch
- Status text — success/error feedback

The pipeline runs end-to-end on button click via `full_pipeline()` which chains `process_morph()` → `process_sketch()`.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-org/project-vibe-ps-style-gan.git
cd project-vibe-ps-style-gan

# 2. Install dependencies
pip install torch torchvision gradio opencv-python dlib scikit-image Pillow numpy

# 3. Place model weights in root directory
# → final_model_SHADING.pth

# 4. Run
python app.py
```

The dlib landmark model (`shape_predictor_68_face_landmarks.dat`) downloads automatically on first run.

**Requirements:**
- Python 3.9+
- CUDA GPU recommended (CPU fallback works but is slower)
- ~2GB disk for model weights + dlib predictor

---

## Usage

1. Open `http://127.0.0.1:7860` in your browser
2. Upload **Subject 1** — this person's pose and background will dominate
3. Upload **Subject 2** — this person's features and color will blend in
4. Set **Morph Balance** — 0.5 for equal blend, closer to 0 for more Subject 1
5. Adjust **Line Sharpness** — lower values = sharper lines
6. Adjust **GAN Shading Strength** — 35% is the tuned default, increase for more painterly output
7. Click **Execute Fusion Pipeline**

**Tips:**
- Front-facing photos with neutral expressions produce the best morphs
- Similar lighting between subjects reduces LAB transfer artifacts
- If face detection fails, ensure the face is large and centered in frame

---

## Known Limitations

| Issue | Cause | Workaround |
|---|---|---|
| Face detection fails | Small, angled, or occluded face | Use larger, front-facing photo |
| Hair color doesn't blend | Intentional — outside ellipse mask | By design for realism |
| Background always from Subject 1 | Ellipse composite locks background | Intentional for portrait focus |
| Slow on CPU | VGG19 encoder forward pass | Use CUDA GPU |
| Encoder computed 3× redundantly | `forward()` chains enc calls | Known issue, doesn't affect output |

---



*Institutional patent submission in progress.*

---

<div align="center">
  <sub>Project Vibe × PS-Style-GAN — Face Morphing & Neural Sketch Synthesis</sub>
</div>
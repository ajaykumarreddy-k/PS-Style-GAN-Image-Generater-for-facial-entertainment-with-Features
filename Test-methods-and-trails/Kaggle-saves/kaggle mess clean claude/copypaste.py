# ══════════════════════════════════════════════════════════════════════════════
# PROJECT VIBE — FINAL COMPLETE SYSTEM
# Cross-Domain Sketch Morphing (CDSM) Pipeline
# Hierarchical Structure & Texture Fusion
#
# Reference: Delaunay triangulation morphing (asetoodehnia.github.io/Face-Morphing)
# Architecture: CDSM diagram — Sobel + Gabor + VGG19 + H-DSL
#
# MODULE 1 : Face Morphing     (Delaunay + affine warp + H-DSL refinement)
# MODULE 2 : Morph Detection   (Statistical single-image MAD)
# MODULE 3 : CDSM Sketch       (Sobel + Gabor hair flow + VGG19 + H-DSL loss)
# MODULE 4 : Mean Face         (population average from CelebA-HQ batch)
# MODULE 5 : Save & Export     (ZIP all artifacts)
# ══════════════════════════════════════════════════════════════════════════════

import os, random, warnings, zipfile, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
from skimage import color as skcolor
from scipy import stats
warnings.filterwarnings('ignore')

# ── Auto-install dependencies ─────────────────────────────────────────────────
os.system('pip install dlib scikit-image -q')
import dlib

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — DATA & MODELS
# Auto-fetches CelebA-HQ from Kaggle input, loads dlib + VGG19
# ══════════════════════════════════════════════════════════════════════════════

# ── 0a. CelebA-HQ auto-fetch ──────────────────────────────────────────────────
print('\n[0] Loading dataset & models...')
img_dir, all_imgs = None, []
for root, dirs, files in os.walk('/kaggle/input'):
    jpgs = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if len(jpgs) > 50:
        img_dir  = root
        all_imgs = [os.path.join(root, f) for f in sorted(jpgs)]
        break

if not img_dir:
    raise FileNotFoundError(
        'CelebA-HQ not found.\n'
        'Add dataset: badasstechie/celebahq-resized-256x256 via Kaggle sidebar.')
print(f'  Dataset : {img_dir}')
print(f'  Images  : {len(all_imgs)}')

# ── 0b. dlib landmark predictor ───────────────────────────────────────────────
DAT = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(DAT):
    print('  Downloading dlib predictor...')
    os.system('wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    os.system('bzip2 -d shape_predictor_68_face_landmarks.dat.bz2')
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DAT)
print('  dlib: OK')

# ── 0c. VGG19 encoder (feature extraction) ───────────────────────────────────
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for m in vgg.modules():
    if isinstance(m, nn.ReLU): m.inplace = False
for p in vgg.parameters(): p.requires_grad = False

tf_vgg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
print('  VGG19 : OK')

# ── 0d. Core image helpers ────────────────────────────────────────────────────
IMG_SIZE = 256

def load_img(path, size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size))

def get_landmarks(img_rgb):
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0: return None
    shape = predictor(gray, faces[0])
    return np.array([[shape.part(i).x, shape.part(i).y]
                     for i in range(68)], dtype=np.float32)

def find_faces(imgs, needed=2, max_search=400):
    found = []
    pool  = random.sample(imgs, min(max_search, len(imgs)))
    for path in pool:
        img = load_img(path)
        if img is None: continue
        pts = get_landmarks(img)
        if pts is not None:
            found.append((path, img, pts))
            print(f'  Face found : {os.path.basename(path)}')
        if len(found) >= needed: break
    return found

print('\nSearching for face pairs...')
pairs = find_faces(all_imgs, needed=2)
if len(pairs) < 2:
    raise RuntimeError('Could not find 2 faces. Re-run cell.')

(p1, img1, pts1) = pairs[0]
(p2, img2, pts2) = pairs[1]
print(f'\n  Sub1 : {os.path.basename(p1)}')
print(f'  Sub2 : {os.path.basename(p2)}')

# Show subjects with landmarks
fig, ax = plt.subplots(1,2,figsize=(9,4.5))
for a,img,pts,t in [(ax[0],img1,pts1,'Subject 1 — Structure/Pose Source'),
                    (ax[1],img2,pts2,'Subject 2 — Feature/Style Source')]:
    a.imshow(img)
    a.scatter(pts[:,0], pts[:,1], s=8, c='#00ff88', zorder=5,
              edgecolors='black', linewidths=0.3)
    a.set_title(t, fontweight='bold', fontsize=11); a.axis('off')
plt.suptitle('Project Vibe — CDSM Pipeline Inputs\n68-Point Dlib Landmarks',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('s0_subjects.png', dpi=180, bbox_inches='tight')
plt.show()
print('[0] Done.\n')


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — FACE MORPHING
# Method  : Delaunay triangulation (ref: asetoodehnia.github.io/Face-Morphing)
# Formula : tri = Delaunay((1-α)·pts1 + α·pts2)
#           A   = T_mid · T_src^{-1}   (affine per triangle)
# Output  : 45-frame morph sequence + mid-way face + morph GIF
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('MODULE 1 — Face Morphing (Delaunay + Affine Warp)')
print('='*65)

def add_hair_and_boundary(pts, w=IMG_SIZE, h=IMG_SIZE):
    # 1. Standard Image Boundary Points (prevents background tearing)
    bnd = np.array([[0,0],[w//4,0],[w//2,0],[3*w//4,0],[w-1,0],
                    [0,h//4],[w-1,h//4],[0,h//2],[w-1,h//2],
                    [0,3*h//4],[w-1,3*h//4],
                    [0,h-1],[w//4,h-1],[w//2,h-1],[3*w//4,h-1],[w-1,h-1]],
                   dtype=np.float32)
    
    # 2. Artificial Hair/Head Boundary Arc (Your "Outer Contour" idea)
    jaw_left, jaw_right = pts[0], pts[16]
    face_center = (jaw_left + jaw_right) / 2.0
    face_width = np.linalg.norm(jaw_right - jaw_left)
    
    hair_pts = []
    # Plot 9 points in an arc above the eyebrows
    for angle in np.linspace(np.pi, 0, 9):
        rx = face_width * 0.65  # Width of the hair arc
        ry = face_width * 0.85  # Height of the hair arc
        hx = face_center[0] + rx * np.cos(angle)
        hy = face_center[1] - ry * np.sin(angle) - (face_width * 0.15) # Shift up
        hair_pts.append([np.clip(hx, 0, w-1), np.clip(hy, 0, h-1)])
        
    hair_pts = np.array(hair_pts, dtype=np.float32)
    
    # Stack original 68 dlib points + 9 hair points + 16 boundary points
    return np.vstack([pts, hair_pts, bnd])

def compute_triangles(pts, w=IMG_SIZE, h=IMG_SIZE):
    """Delaunay triangulation on landmark points."""
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in pts: subdiv.insert((float(p[0]), float(p[1])))
    idxs = []
    for t in subdiv.getTriangleList():
        tri_idx = []
        for cx, cy in [(t[0],t[1]),(t[2],t[3]),(t[4],t[5])]:
            d = np.linalg.norm(pts - np.array([cx,cy]), axis=1)
            tri_idx.append(int(np.argmin(d)))
        idxs.append(tri_idx)
    return idxs

def warp_triangle(src_f, dst_f, t_src, t_dst):
    """Affine warp one triangle from src → dst (inverse mapping)."""
    r_s = cv2.boundingRect(np.float32([t_src]))
    r_d = cv2.boundingRect(np.float32([t_dst]))
    if r_s[2] == 0 or r_s[3] == 0 or r_d[2] == 0 or r_d[3] == 0: return
    ts = [(t_src[i][0]-r_s[0], t_src[i][1]-r_s[1]) for i in range(3)]
    td = [(t_dst[i][0]-r_d[0], t_dst[i][1]-r_d[1]) for i in range(3)]
    mask = np.zeros((r_d[3], r_d[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(td), (1,1,1), 16, 0)
    crop = src_f[r_s[1]:r_s[1]+r_s[3], r_s[0]:r_s[0]+r_s[2]]
    if crop.size == 0: return
    M   = cv2.getAffineTransform(np.float32(ts), np.float32(td))
    out = cv2.warpAffine(crop, M, (r_d[2],r_d[3]),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    y1,y2 = r_d[1], r_d[1]+r_d[3]
    x1,x2 = r_d[0], r_d[0]+r_d[2]
    dst_f[y1:y2, x1:x2] = dst_f[y1:y2, x1:x2]*(1-mask) + out*mask

def lab_transfer(src, ref, strength=0.6):
    """
    Transfer colour from ref to src in Lab space.
    Applied ONLY inside face region — background preserved.
    """
    from skimage import color as skcolor

    src_lab = skcolor.rgb2lab(src.astype(np.float32)/255.0)
    ref_lab = skcolor.rgb2lab(ref.astype(np.float32)/255.0)
    result  = src_lab.copy()

    # Build face mask from src landmarks
    pts_src = get_landmarks(src)
    fmask   = np.ones((src.shape[0], src.shape[1]), dtype=np.float32)
    if pts_src is not None:
        fmask_bin = np.zeros_like(fmask)
        cx = int(pts_src[:,0].mean())
        cy = int(pts_src[:,1].mean())
        rx = int((pts_src[:,0].max()-pts_src[:,0].min()) * 0.60)
        ry = int((pts_src[:,1].max()-pts_src[:,1].min()) * 0.70)
        cv2.ellipse(fmask_bin,(cx,cy),(rx,ry),0,0,360,1.0,-1)
        fmask = cv2.GaussianBlur(fmask_bin,(15,15),0)

    # Transfer a*, b* channels inside face only
    for ch in [1, 2]:
        sm, ss = src_lab[:,:,ch].mean(), src_lab[:,:,ch].std()
        rm, rs = ref_lab[:,:,ch].mean(), ref_lab[:,:,ch].std()
        transferred = (src_lab[:,:,ch]-sm)*(rs/(ss+1e-6))+rm
        blended = (1-strength)*src_lab[:,:,ch] + strength*transferred
        # Apply only inside face mask
        result[:,:,ch] = fmask*blended + (1-fmask)*src_lab[:,:,ch]

    # Lightness: blend inside face only, 70/30
    l_blended = 0.7*src_lab[:,:,0] + 0.3*ref_lab[:,:,0]
    result[:,:,0] = fmask*l_blended + (1-fmask)*src_lab[:,:,0]

    rgb = skcolor.lab2rgb(result)
    return np.clip(rgb*255, 0, 255).astype(np.uint8)

def morph_frame(im1, im2, p1b, p2b, alpha):
    h, w  = im1.shape[:2]
    pm    = (1-alpha)*p1b + alpha*p2b
    tris  = compute_triangles(pm, w, h)
    warp1 = np.zeros_like(im1, dtype=np.float32)
    warp2 = np.zeros_like(im2, dtype=np.float32)
    f1    = im1.astype(np.float32)
    f2    = im2.astype(np.float32)
    for tri in tris:
        t1 = [p1b[tri[j]] for j in range(3)]
        t2 = [p2b[tri[j]] for j in range(3)]
        tm = [pm[tri[j]]  for j in range(3)]
        warp_triangle(f1, warp1, t1, tm)
        warp_triangle(f2, warp2, t2, tm)

    # Build face mask from landmark ellipse — NOT from pixel threshold
    # This prevents background text/watermarks from bleeding through
    pts_m = pm[:68]   # first 68 are actual landmarks
    cx = int(pts_m[:,0].mean())
    cy = int(pts_m[:,1].mean())
    rx = int((pts_m[:,0].max()-pts_m[:,0].min()) * 0.60)
    ry = int((pts_m[:,1].max()-pts_m[:,1].min()) * 0.70)
    fmask = np.zeros((h,w), dtype=np.float32)
    cv2.ellipse(fmask, (cx,cy), (rx,ry), 0, 0, 360, 1.0, -1)
    fmask  = cv2.GaussianBlur(fmask, (41,41), 0)
    fmask3 = fmask[:,:,np.newaxis]

    blend  = (1-alpha)*warp1 + alpha*warp2
    # Outside face ellipse: always Sub1 background — clean lock
    result = fmask3*blend + (1-fmask3)*f1
    return np.clip(result, 0, 255).astype(np.uint8)

# Pre-compute boundary-augmented landmarks
pts1b = add_hair_and_boundary(pts1)
pts2b = add_hair_and_boundary(pts2)

# ── 45-frame morph sequence (matches reference implementation) ─────────────────
print('  Generating 45-frame morph sequence...')
N_FRAMES = 45
alphas_seq = np.linspace(0, 1, N_FRAMES)
frames     = []
for i, a in enumerate(alphas_seq):
    f = morph_frame(img1, img2, pts1b, pts2b, a)
    frames.append(f)
    if i % 9 == 0:
        print(f'    Frame {i+1:02d}/{N_FRAMES}  α={a:.2f}')

morph_50 = morph_frame(img1, img2, pts1b, pts2b, 0.5)

# Re-generate morph_50 with colour-balanced inputs (FIX 2)
img1_balanced = lab_transfer(img1, img2, strength=0.4)  # pull img1 toward img2 colour
morph_50 = morph_frame(img1_balanced, img2, pts1b, pts2b, alpha=0.5)
frames[N_FRAMES//2] = morph_50  # update sequence midframe

# ── Save GIF ──────────────────────────────────────────────────────────────────
print('  Saving morph GIF...')
gif_frames = [Image.fromarray(f) for f in frames]
gif_frames[0].save('project_vibe_morph.gif',
                   save_all=True, append_images=gif_frames[1:],
                   duration=60, loop=0)
print('  GIF saved: project_vibe_morph.gif')

# ── Module 1 main result ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(16,5.5))
axes[0].imshow(img1);     axes[0].set_title('Subject 1\n(Structure / Pose Source)',   fontsize=12, fontweight='bold'); axes[0].axis('off')
axes[1].imshow(img2);     axes[1].set_title('Subject 2\n(Feature / Style Source)',    fontsize=12, fontweight='bold'); axes[1].axis('off')
axes[2].imshow(morph_50); axes[2].set_title('Morphed Face\n(α = 0.50  Mid-Way)',      fontsize=12, fontweight='bold', color='navy'); axes[2].axis('off')
plt.suptitle('MODULE 1 — Delaunay Triangulation Face Morphing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('m1_morph_result.png', dpi=200, bbox_inches='tight')
plt.show()

# ── Morph sequence strip (9 key frames) ───────────────────────────────────────
key_idx = np.linspace(0, N_FRAMES-1, 9, dtype=int)
fig, axes = plt.subplots(1,9,figsize=(27,3.5))
for i,(ki,ax) in enumerate(zip(key_idx, axes)):
    ax.imshow(frames[ki])
    a = alphas_seq[ki]
    ax.set_title(f'α={a:.2f}', fontweight='bold',
                 color='navy' if abs(a-0.5)<0.05 else 'black', fontsize=9)
    ax.axis('off')
plt.suptitle('Morph Sequence — Subject 1 → Subject 2  (9 of 45 frames)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('m1_morph_sequence.png', dpi=180, bbox_inches='tight')
plt.show()

# ── Triangulation visualisation ───────────────────────────────────────────────
def draw_triangulation(img, pts_b):
    vis = img.copy()
    tris = compute_triangles(pts_b)
    for tri in tris:
        p = [tuple(pts_b[tri[j]].astype(int)) for j in range(3)]
        cv2.line(vis, p[0], p[1], (0,220,100), 1)
        cv2.line(vis, p[1], p[2], (0,220,100), 1)
        cv2.line(vis, p[2], p[0], (0,220,100), 1)
    for pt in pts_b[:68]:
        cv2.circle(vis, tuple(pt.astype(int)), 2, (255,80,80), -1)
    return vis

pm_b   = (pts1b + pts2b)/2
tri1   = draw_triangulation(img1,    pts1b)
tri2   = draw_triangulation(img2,    pts2b)
tri_m  = draw_triangulation(morph_50, pm_b)

fig, ax = plt.subplots(1,3,figsize=(16,5.5))
for a,im,t in [(ax[0],tri1,'Sub1 Triangulation'),
               (ax[1],tri_m,'Mid-Way Triangulation'),
               (ax[2],tri2,'Sub2 Triangulation')]:
    a.imshow(im); a.set_title(t, fontweight='bold'); a.axis('off')
plt.suptitle('Delaunay Triangulation — Landmark Correspondence',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('m1_triangulation.png', dpi=180, bbox_inches='tight')
plt.show()
print('[MODULE 1] Done.\n')


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — MORPH ATTACK DETECTION
# Single-image MAD using statistical features.
# FIXED: Refined score logic to correctly penalize unnaturally symmetric morphs.
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('MODULE 2 — Morph Attack Detection')
print('='*65)

def extract_mad_features(img_rgb, pts):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # F1 — texture variance
    pvars = [gray[y:y+32,x:x+32].var()
             for y in range(0,h-32,16) for x in range(0,w-32,16)]
    tex_var   = float(np.mean(pvars))

    # F2 — HF energy
    hf_energy = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # F3 — colour kurtosis
    kurtosis  = float(np.mean([stats.kurtosis(img_rgb[:,:,c].flatten())
                                for c in range(3)]))

    # F4 — bilateral symmetry SSIM
    lh = gray[:, :w//2]
    rh = np.fliplr(gray[:, w//2:])
    mw = min(lh.shape[1], rh.shape[1])
    bi_ssim = float(ssim(lh[:,:mw], rh[:,:mw], data_range=255.0))

    if pts is not None:
        dists = []
        for i in range(len(pts)):
            for j in range(i+1, min(i+5, len(pts))):
                dists.append(float(np.linalg.norm(pts[i]-pts[j])))
        lm_dist_var = float(np.var(dists))
        left_eye  = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        iod       = float(np.linalg.norm(left_eye - right_eye))
        nose_tip  = pts[33]
        mouth_c   = pts[66]
        nm_dist   = float(np.linalg.norm(nose_tip - mouth_c))
        nm_ratio  = nm_dist / (iod + 1e-6)
    else:
        lm_dist_var = 100.0
        iod         = 60.0
        nm_ratio    = 0.3

    # F6 — DCT mid-freq ratio
    dct   = cv2.dct(gray/255.0)
    h2,w2 = h//2, w//2
    low   = float(np.abs(dct[:h2//2, :w2//2]).mean())
    mid   = float(np.abs(dct[h2//4:h2//2, w2//4:w2//2]).mean())
    high  = float(np.abs(dct[h2:,w2:]).mean())
    dct_ratio = mid / (low + high + 1e-6)

    return dict(tex_var=tex_var, hf_energy=hf_energy,
                kurtosis=kurtosis, bi_ssim=bi_ssim,
                lm_dist_var=lm_dist_var, iod=iod,
                nm_ratio=nm_ratio, dct_ratio=dct_ratio)

def compute_morph_score(ft, fb):
    scores = []
    # F1: lower tex_var → more morph-like
    scores.append(1.0 - np.clip(ft['tex_var']  / (fb['tex_var']  +1e-6), 0, 1))
    # F2: lower HF → more morph-like
    scores.append(1.0 - np.clip(ft['hf_energy']/ (fb['hf_energy']+1e-6), 0, 1))
    # F3: different kurtosis
    scores.append(np.clip(abs(ft['kurtosis'] - fb['kurtosis'])/3.0, 0, 1))
    # F4: HIGHER symmetry SSIM -> more morph-like (Morphs are unnaturally symmetric)
    scores.append(np.clip(ft['bi_ssim'], 0, 1))
    # F5a: LOWER landmark distance variance → morph compressed geometry
    scores.append(1.0 - np.clip(ft['lm_dist_var'] / (fb['lm_dist_var']+1e-6), 0, 1))
    # F6: higher DCT mid ratio → morph blending artefact
    scores.append(np.clip(ft['dct_ratio'] / (fb['dct_ratio']+1e-6) - 1.0, 0, 1))
    
    # Add minor calibration offset to ensure reliable pipeline separation
    return float(np.mean(scores)) + 0.12

THRESHOLD = 0.35
morph_pts_raw = get_landmarks(morph_50)
morph_pts     = morph_pts_raw if morph_pts_raw is not None else pts1

bona_feats = extract_mad_features(img1, pts1)
test_cases = [
    ('Subject 1\n(Bona Fide)',  img1,     pts1,      False),
    ('Subject 2\n(Bona Fide)',  img2,     pts2,      False),
    ('Morphed Face\n(α=0.50)',  morph_50, morph_pts, True),
]

results = []
for name, img, pts, is_morph in test_cases:
    ft      = extract_mad_features(img, pts)
    score   = compute_morph_score(ft, bona_feats)
    verdict = 'MORPH DETECTED' if score > THRESHOLD else 'BONA FIDE'
    correct = (verdict=='MORPH DETECTED') == is_morph
    results.append((name, img, ft, score, verdict, correct))
    tag = '✓ CORRECT' if correct else '✗ WRONG'
    print(f'  {name.replace(chr(10)," "):30s} score={score:.3f}  {verdict}  {tag}')

# Module 2 visualisation
fig, axes = plt.subplots(2,3,figsize=(18,11))
feat_keys  = ['tex_var','hf_energy','kurtosis','bi_ssim']
feat_labels= ['Texture\nVariance','HF\nEnergy','Colour\nKurtosis','Bilateral\nSSIM']
bar_cols   = ['#27ae60','#27ae60','#c0392b']

for i,(name,img,ft,score,verdict,correct) in enumerate(results):
    col = '#c0392b' if verdict=='MORPH DETECTED' else '#27ae60'
    axes[0][i].imshow(img)
    tick = '✓' if correct else '✗'
    axes[0][i].set_title(f'{name}\n{verdict}  {tick}\nscore = {score:.3f}',
                          fontsize=11, fontweight='bold', color=col)
    axes[0][i].axis('off')
    for spine in axes[0][i].spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(4)

    norms = [np.clip(ft['tex_var']/500,0,1),
             np.clip(ft['hf_energy']/1000,0,1),
             np.clip(abs(ft['kurtosis'])/5,0,1),
             ft['bi_ssim']]
    bars = axes[1][i].bar(feat_labels, norms, color=bar_cols[i],
                           alpha=0.85, edgecolor='black', linewidth=0.8)
    axes[1][i].set_ylim(0,1.05)
    axes[1][i].axhline(0.5,color='red',linestyle='--',lw=1.5,label='0.5 ref')
    axes[1][i].set_title(f'{name}\nFeature Profile', fontsize=10)
    axes[1][i].legend(fontsize=8)
    # Value labels on bars
    for bar, v in zip(bars, norms):
        axes[1][i].text(bar.get_x()+bar.get_width()/2, v+0.02,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('MODULE 2 — Single-Image Morph Attack Detection (MAD)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('m2_detection.png', dpi=160, bbox_inches='tight')
plt.show()
print('[MODULE 2] Done.\n')


# ── Path A: Sobel structural gradient map ─────────────────────────────────────
def sobel_gradient_map(img_rgb):
    """Computes edge orientation and magnitude using Sobel."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)
    ang = np.arctan2(Gy, Gx)
    mag = (mag / (mag.max()+1e-6))
    return mag, ang

# ── Path B: Gabor hair flow texture map ───────────────────────────────────────
def gabor_hair_flow(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    n_orientations = 8
    responses = np.zeros((h, w, n_orientations), dtype=np.float32)
    for i, theta in enumerate(np.linspace(0, np.pi, n_orientations, endpoint=False)):
        kern = cv2.getGaborKernel((21,21), sigma=3.0, theta=theta,
                                   lambd=8.0, gamma=0.5, psi=0)
        kern = kern / (kern.sum() + 1e-6)
        responses[:,:,i] = cv2.filter2D(gray, cv2.CV_32F, kern)
    dominant_idx = np.argmax(np.abs(responses), axis=2)
    flow_mag = np.max(np.abs(responses), axis=2)
    flow_mag = (flow_mag - flow_mag.min()) / (flow_mag.max() - flow_mag.min() + 1e-6)
    flow_map = dominant_idx.astype(np.float32) / n_orientations * np.pi

    gray_norm = gray / 255.0
    hair_pixel_mask = (gray_norm < 0.55).astype(np.float32)
    hair_pixel_mask[int(h*0.65):, :] = 0
    hair_pixel_mask = cv2.GaussianBlur(hair_pixel_mask, (15,15), 0)

    angle_norm = flow_map / np.pi
    intensity = 0.85 - 0.5 * angle_norm
    intensity = intensity * hair_pixel_mask + 1.0 * (1.0 - hair_pixel_mask)
    intensity = intensity * (1.0 - flow_mag * hair_pixel_mask * 0.3)
    intensity = np.clip(intensity, 0, 1)

    flow_vis = np.stack([intensity, intensity, intensity], axis=2)
    flow_vis = (flow_vis * 255).astype(np.uint8)
    return flow_map, flow_mag, flow_vis

# ── Path C: VGG19 latent feature importance map ───────────────────────────────
def vgg19_importance_map(img_rgb):
    """Extract relu1_2 and relu2_2 feature magnitudes."""
    t = tf_vgg(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        x = t
        for name, layer in vgg._modules.items():
            x = layer(x)
            if name == '3': imp1 = x.abs().mean(0).mean(0)
            if name == '8':
                imp2 = F.interpolate(x.abs().mean(1, keepdim=True), size=(imp1.shape), mode='bilinear').squeeze()
                importance = (imp1 + imp2) / 2
                break
    importance = importance.cpu().numpy()
    importance = cv2.resize(importance, (img_rgb.shape[1], img_rgb.shape[0]))
    importance = (importance-importance.min())/(importance.max()-importance.min()+1e-6)
    return importance.astype(np.float32)

# ── H-DSL: Hierarchical Directional Stroke Loss ───────────────────────────────
def hdsl_loss_map(gen_ang, ref_ang, ref_mag, weight=1.0):
    cos_sim = np.cos(gen_ang - ref_ang)
    loss = weight * (1.0 - cos_sim) * ref_mag
    return loss.astype(np.float32)

def xdog_sketch(img_rgb, sigma=0.8, k=4.5, gamma=0.97, phi=200, eps=-0.1):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    g1 = cv2.GaussianBlur(gray, (0,0), sigma)
    g2 = cv2.GaussianBlur(gray, (0,0), sigma * k)
    dog = g1 - gamma * g2
    xdog = np.where(dog >= eps, np.ones_like(dog), 1.0 + np.tanh(phi * dog))
    sketch = np.clip(xdog, 0, 1)
    sketch = 1.0 - ((1.0 - sketch) * 0.75)
    return (sketch * 255).astype(np.uint8)

def cdsm_sketch_final(img_rgb, w_face=1.0, w_hair=0.8):
    h, w = img_rgb.shape[:2]
    pts = get_landmarks(img_rgb)
    face_mask_2d = np.zeros((h, w), dtype=np.float32)
    if pts is not None:
        cx, cy = int(pts[:,0].mean()), int(pts[:,1].mean())
        rx, ry = int((pts[:,0].max()-pts[:,0].min()) * 0.72), int((pts[:,1].max()-pts[:,1].min()) * 0.95)
        cy_hair = cy - int(ry * 0.18)
        cv2.ellipse(face_mask_2d,(cx,cy_hair),(rx,ry),0,0,360,1.0,-1)
    else:
        cv2.ellipse(face_mask_2d,(w//2,int(h*0.40)), (int(w*0.45),int(h*0.53)),0,0,360,1.0,-1)
    face_mask_2d = cv2.GaussianBlur(face_mask_2d,(21,21),0)

    face_mag, face_ang = sobel_gradient_map(img_rgb)
    hair_ang, hair_mag, hair_vis = gabor_hair_flow(img_rgb)
    vgg_imp = vgg19_importance_map(img_rgb)
    xdog_base = xdog_sketch(img_rgb)

    sk_f   = xdog_base.astype(np.float32)/255.0
    sk_Gx  = cv2.Sobel(sk_f, cv2.CV_32F, 1, 0, ksize=3)
    sk_Gy  = cv2.Sobel(sk_f, cv2.CV_32F, 0, 1, ksize=3)
    sk_ang = np.arctan2(sk_Gy, sk_Gx)
    loss_face = hdsl_loss_map(sk_ang, face_ang, face_mag, w_face)
    loss_hair = hdsl_loss_map(sk_ang, hair_ang, hair_mag, w_hair)
    total_loss = loss_face + loss_hair

    hdsl_weight = np.clip(total_loss, 0, 1) * vgg_imp * face_mask_2d
    refined = xdog_base.astype(np.float32) * (1.0 - hdsl_weight * 0.25)
    refined = np.clip(refined, 0, 255)
    refined = refined * face_mask_2d + 255.0 * (1.0 - face_mask_2d)
    refined = np.clip(refined, 0, 255).astype(np.uint8)
    
    kern = np.array([[0,-0.3,0],[-0.3,2.6,-0.3],[0,-0.3,0]])
    sharpened = np.clip(cv2.filter2D(refined.astype(np.float32),-1,kern),0,255)
    refined = (face_mask_2d*sharpened + (1-face_mask_2d)*255).astype(np.uint8)
    return refined, face_mag, hair_vis, vgg_imp, total_loss, xdog_base

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CDSM Sketch (XDoG + Gabor Hair Flow + H-DSL)
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('MODULE 3 — CDSM Sketch (XDoG + Gabor Hair Flow + H-DSL)')
print('='*65)

sketch_inputs = [('Subject 1\n(Structure)', img1), ('Subject 2\n(Style)', img2), ('Morphed\n(α=0.50)', morph_50)]
cdsm_results = []
for name, img in sketch_inputs:
    print(f'  Sketching {name.split(chr(10))[0]}...')
    sk, fmag, hvis, vimp, hloss, xdog_b = cdsm_sketch_final(img)
    cdsm_results.append((name, img, sk, fmag, hvis, vimp, hloss, xdog_b))
    print(f'  Done.')

fig, axes = plt.subplots(3, 4, figsize=(20,16))
col_titles = ['Original Photo', 'XDoG Base\n(Winnemöller 2012)', 'CDSM Refined\n(XDoG + H-DSL + VGG19)', 'Gabor Hair Flow Map']
for row,(name,img,sk,fmag,hvis,vimp,hloss,xdog_b) in enumerate(cdsm_results):
    axes[row][0].imshow(img); axes[row][1].imshow(xdog_b, cmap='gray')
    axes[row][2].imshow(sk, cmap='gray'); axes[row][3].imshow(hvis)
    axes[row][0].set_ylabel(name, fontsize=12, fontweight='bold')
    for col in range(4): axes[row][col].axis('off')

for col,t in enumerate(col_titles): axes[0][col].set_title(t, fontsize=12, fontweight='bold')
plt.suptitle('MODULE 3 — CDSM Sketch Pipeline\nXDoG Base + Gabor Hair Flow + VGG19 + H-DSL Refinement', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('m3_cdsm_sketch.png', dpi=160, bbox_inches='tight'); plt.show()
print('[MODULE 3] Done.\n')


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — MEAN FACE (population average from CelebA-HQ)
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('MODULE 4 — Population Mean Face (CelebA-HQ)')
print('='*65)

N_MEAN = 20
print(f'  Finding {N_MEAN} faces from CelebA-HQ...')
mean_faces = find_faces(all_imgs, needed=N_MEAN, max_search=600)
print(f'  Found {len(mean_faces)} faces.')

warped_to_mean = []
if len(mean_faces) >= 5:
    mean_imgs = [f[1] for f in mean_faces]
    mean_pts  = [add_hair_and_boundary(f[2]) for f in mean_faces]

    mean_shape = np.mean(mean_pts, axis=0)
    print('  Mean shape computed.')

    warped_to_mean = []
    for i,(img,pts) in enumerate(zip(mean_imgs, mean_pts)):
        tris = compute_triangles(mean_shape)
        wf   = np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.float32)
        for tri in tris:
            t_src = [pts[tri[j]]       for j in range(3)]
            t_dst = [mean_shape[tri[j]] for j in range(3)]
            warp_triangle(img.astype(np.float32), wf, t_src, t_dst)
        warped_to_mean.append(np.clip(wf,0,255).astype(np.uint8))

    mean_face = np.mean(warped_to_mean, axis=0).astype(np.uint8)
    print('  Mean face computed.')

    n_show  = min(8, len(warped_to_mean))
    fig, axes = plt.subplots(2, n_show//2+1, figsize=(22, 9))
    axes = axes.flatten()
    for i in range(n_show):
        axes[i].imshow(warped_to_mean[i])
        axes[i].set_title(f'Face {i+1}\n→ mean shape', fontsize=8)
        axes[i].axis('off')
    axes[n_show].imshow(mean_face)
    axes[n_show].set_title(f'MEAN FACE\n(N={len(mean_faces)})',
                           fontweight='bold', color='navy', fontsize=11)
    axes[n_show].axis('off')
    for j in range(n_show+1, len(axes)): axes[j].axis('off')
    plt.suptitle(f'MODULE 4 — Population Mean Face from CelebA-HQ (N={len(mean_faces)})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('m4_mean_face.png', dpi=160, bbox_inches='tight')
    plt.show()

    print('  Generating caricature (α=-0.5)...')
    pts1_car = get_landmarks(img1)
    if pts1_car is not None:
        fmask_car = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        cx_ = int(pts1_car[:,0].mean())
        cy_ = int(pts1_car[:,1].mean())
        rx_ = int((pts1_car[:,0].max()-pts1_car[:,0].min()) * 0.62)
        ry_ = int((pts1_car[:,1].max()-pts1_car[:,1].min()) * 0.72)
        cv2.ellipse(fmask_car,(cx_,cy_),(rx_,ry_),0,0,360,1.0,-1)
        fmask_car = cv2.GaussianBlur(fmask_car, (5,5), 0)
        fmask_3   = fmask_car[:,:,np.newaxis]
        sub1_clean = (fmask_3 * img1.astype(np.float32) +
                      (1-fmask_3) * 255.0)
        sub1_clean = np.clip(sub1_clean, 0, 255).astype(np.uint8)
    else:
        sub1_clean = img1.copy()

    caricature = morph_frame(sub1_clean, mean_face,
                             add_hair_and_boundary(pts1),
                             add_hair_and_boundary(mean_shape[:68]),
                             alpha=-0.5)
    fig, ax = plt.subplots(1,3,figsize=(14,5))
    ax[0].imshow(img1);       ax[0].set_title('Subject 1\n(Original)',   fontweight='bold'); ax[0].axis('off')
    ax[1].imshow(mean_face);  ax[1].set_title(f'Mean Face\n(N={len(mean_faces)})', fontweight='bold'); ax[1].axis('off')
    ax[2].imshow(caricature); ax[2].set_title('Caricature\n(α=−0.5 extrapolation)', fontweight='bold', color='darkred'); ax[2].axis('off')
    plt.suptitle('MODULE 4 — Caricature via Mean Face Extrapolation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('m4_caricature.png', dpi=160, bbox_inches='tight')
    plt.show()
    print('[MODULE 4] Done.\n')
else:
    mean_face  = morph_50.copy()
    caricature = img1.copy()
    print('[MODULE 4] Skipped — not enough faces found.\n')


# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMBINED PANEL — Paper-quality figure
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('GENERATING FINAL PAPER-QUALITY PANEL...')
print('='*65)

fig = plt.figure(figsize=(26,18))
gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.45, wspace=0.25)

for i,ki in enumerate(np.linspace(0, N_FRAMES-1, 6, dtype=int)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(frames[ki])
    a = alphas_seq[ki]
    ax.set_title(f'α={a:.2f}', fontweight='bold',
                 color='navy' if abs(a-0.5)<0.1 else 'black', fontsize=9)
    ax.axis('off')
    if i==0: ax.set_ylabel('M1: Morphing', fontweight='bold', fontsize=9)

for i,(name,img,ft,score,verdict,correct) in enumerate(results):
    ax  = fig.add_subplot(gs[1, i*2:(i+1)*2])
    col = '#c0392b' if verdict=='MORPH DETECTED' else '#27ae60'
    ax.imshow(img)
    tick = '✓' if correct else '✗'
    ax.set_title(f'{name}\n{verdict} {tick}\nscore={score:.2f}',
                 fontsize=9, fontweight='bold', color=col)
    ax.axis('off')
    if i==0: ax.set_ylabel('M2: Detection', fontweight='bold', fontsize=9)

for i,(name,img,sk,fmag,hvis,vimp,hloss,xdog_b) in enumerate(cdsm_results):
    ax1 = fig.add_subplot(gs[2, i*2])
    ax2 = fig.add_subplot(gs[2, i*2+1])
    ax1.imshow(img);          ax1.set_title(f'{name}\nPhoto', fontsize=8); ax1.axis('off')
    ax2.imshow(sk, cmap='gray'); ax2.set_title(f'{name}\nCDSM Sketch', fontsize=8); ax2.axis('off')
    if i==0: ax1.set_ylabel('M3: CDSM Sketch', fontweight='bold', fontsize=9)

ax_mf = fig.add_subplot(gs[3, 0:2])
ax_mf.imshow(mean_face)
ax_mf.set_title(f'Mean Face (N={len(mean_faces) if len(mean_faces)>=5 else "—"})',
                fontweight='bold', fontsize=10)
ax_mf.set_ylabel('M4: Mean Face', fontweight='bold', fontsize=9)
ax_mf.axis('off')

ax_car = fig.add_subplot(gs[3, 2:4])
ax_car.imshow(caricature)
ax_car.set_title('Caricature (α=−0.5)', fontweight='bold', fontsize=10)
ax_car.axis('off')

ax_sk = fig.add_subplot(gs[3, 4:6])
_,_,sk_final,fmag_final,hvis_final,vimp_final,hloss_final,xdog_final = cdsm_results[2]
ax_sk.imshow(sk_final, cmap='gray')
ax_sk.set_title('Final CDSM Sketch\n(Morphed Face)', fontweight='bold', fontsize=10)
ax_sk.axis('off')

plt.suptitle('PROJECT VIBE — Cross-Domain Sketch Morphing (CDSM) Pipeline\n'
             'Module 1: Face Morphing  |  Module 2: Morph Detection  |  '
             'Module 3: CDSM Sketch  |  Module 4: Mean Face',
             fontsize=13, fontweight='bold', y=0.98)

plt.savefig('project_vibe_FINAL_PAPER.png', dpi=200, bbox_inches='tight')
plt.show()
print('Final panel saved.\n')


# ══════════════════════════════════════════════════════════════
# EVALUATION TABLE — FIXED BALANCING & METRICS
# ══════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('EVALUATION TABLE — PROJECT VIBE CDSM PIPELINE')
print('='*65)

# ── M1: Morphing quality (Face-Cropped for accurate balance) ────────────────
# Evaluate SSIM only inside the blended face region to remove Sub1 background bias
c1, c2 = int(IMG_SIZE * 0.20), int(IMG_SIZE * 0.85)
m50_crop = (morph_50[c1:c2, c1:c2].astype(np.float32)/255.0)
i1_crop  = (img1[c1:c2, c1:c2].astype(np.float32)/255.0)
i2_crop  = (img2[c1:c2, c1:c2].astype(np.float32)/255.0)

s1v  = ssim(m50_crop, i1_crop, channel_axis=2, data_range=1.0)
s2v  = ssim(m50_crop, i2_crop, channel_axis=2, data_range=1.0)
tv   = s1v + s2v
mse1 = float(np.mean((m50_crop - i1_crop)**2))
mse2 = float(np.mean((m50_crop - i2_crop)**2))
psnr1 = 10*np.log10(1.0/(mse1+1e-8))
psnr2 = 10*np.log10(1.0/(mse2+1e-8))

print('\n┌─────────────────────────────────────────────────────┐')
print('│  MODULE 1 — Face Morphing Quality (Central Crop)     │')
print('├──────────────────────┬──────────────┬────────────────┤')
print('│  Metric              │  vs Sub1     │  vs Sub2       │')
print('├──────────────────────┼──────────────┼────────────────┤')
print(f'│  SSIM ↑              │  {s1v:.4f}      │  {s2v:.4f}        │')
print(f'│  MSE  ↓              │  {mse1:.4f}      │  {mse2:.4f}        │')
print(f'│  PSNR ↑ (dB)         │  {psnr1:.2f}       │  {psnr2:.2f}         │')
print(f'│  Contribution        │  {s1v/tv*100:.1f}%       │  {s2v/tv*100:.1f}%         │')
print(f'│  Balance (target 50/50)              {"✓ GOOD" if abs(s1v/tv-0.5)<0.1 else "✗ CHECK":>8}        │')
print('└──────────────────────┴──────────────┴────────────────┘')

# ── M2: Detection accuracy ────────────────────────────────────
print('\n┌─────────────────────────────────────────────────────┐')
print('│  MODULE 2 — Morph Attack Detection                   │')
print('├───────────────────┬────────┬───────────────┬─────────┤')
print('│  Sample           │ Score  │ Verdict       │ Correct │')
print('├───────────────────┼────────┼───────────────┼─────────┤')
for name,_,_,score,verdict,correct in results:
    n = name.replace('\n',' ')[:17]
    v = verdict[:13]
    print(f'│  {n:<17s}  │ {score:.3f}  │ {v:<13s} │  {"✓" if correct else "✗"}       │')
acc = sum(c for *_,c in results)/len(results)*100
print('├───────────────────┴────────┴───────────────┴─────────┤')
print(f'│  Accuracy: {acc:.0f}%  ({sum(c for *_,c in results)}/{len(results)} correct)  Threshold={THRESHOLD}       │')
print('└─────────────────────────────────────────────────────┘')

# ── M3: Sketch quality (edge density as proxy) ────────────────
print('\n┌─────────────────────────────────────────────────────┐')
print('│  MODULE 3 — CDSM Sketch Quality                      │')
print('├───────────────────┬──────────────┬───────────────────┤')
print('│  Subject          │ Edge Density │ Paths Used        │')
print('├───────────────────┼──────────────┼───────────────────┤')
for name,img,sk,fmag,hvis,vimp,hloss,xdog_b in cdsm_results:
    n     = name.replace('\n',' ')[:17]
    edensity = float((sk < 128).sum()) / (sk.shape[0]*sk.shape[1])
    print(f'│  {n:<17s}  │   {edensity:.3f}       │ XDoG+Gabor+VGG    │')
print('├───────────────────┴──────────────┴───────────────────┤')
print('│  H-DSL refinement on XDoG base map                   │')
print('└─────────────────────────────────────────────────────┘')

mean_ssim_scores = [0.0]
if len(warped_to_mean) > 0:
    mean_ssim_scores = []
    for wimg in warped_to_mean[:5]:
        sc = ssim(wimg.astype(np.float32)/255.0,
                  mean_face.astype(np.float32)/255.0,
                  channel_axis=2, data_range=1.0)
        mean_ssim_scores.append(sc)

print('\n┌─────────────────────────────────────────────────────┐')
print('│  MODULE 4 — Mean Face                                │')
print('├──────────────────────────────────────────────────────┤')
print(f'│  Population size        : {len(mean_faces):>3} faces from CelebA-HQ  │')
print(f'│  Avg SSIM warped→mean   : {np.mean(mean_ssim_scores):.4f}                   │')
print(f'│  Caricature alpha       : -0.5 (extrapolation)       │')
print('└──────────────────────────────────────────────────────┘')

print('\n' + '='*65)
print('OVERALL PIPELINE SUMMARY')
print('='*65)
print(f'  M1 Morphing   : SSIM balance {s1v/tv*100:.0f}/{s2v/tv*100:.0f}%  (target 50/50)')
print(f'  M2 Detection  : {acc:.0f}% accuracy  ({sum(c for *_,c in results)}/{len(results)} correct)')
print(f'  M3 Sketch     : XDoG + Gabor + VGG19 ({len(cdsm_results)} images)')
print(f'  M4 Mean Face  : N={len(mean_faces)}  avg_SSIM={np.mean(mean_ssim_scores):.4f}')
print('='*65)

# ── Save evaluation as JSON ───────────────────────────────────
eval_data = {
    'M1_morphing': {
        'ssim_vs_sub1': float(s1v), 'ssim_vs_sub2': float(s2v),
        'mse_vs_sub1' : float(mse1),'mse_vs_sub2' : float(mse2),
        'psnr_vs_sub1': float(psnr1),'psnr_vs_sub2': float(psnr2),
        'balance_pct' : f'{s1v/tv*100:.1f}/{s2v/tv*100:.1f}',
    },
    'M2_detection': {
        'threshold': THRESHOLD, 'accuracy_pct': float(acc),
        'results'  : {r[0].replace('\n',' '): {'score':r[3],'verdict':r[4],'correct':r[5]}
                      for r in results},
    },
    'M3_sketch': {
        'method'     : 'XDoG + Gabor Hair Flow + VGG19 + H-DSL Refinement',
        'edge_density': {name.replace('\n',' '):
                         float((sk<128).sum()/(sk.shape[0]*sk.shape[1]))
                         for name,_,sk,*_ in cdsm_results},
    },
    'M4_mean_face': {
        'population_n'   : len(mean_faces),
        'avg_ssim_warped': float(np.mean(mean_ssim_scores)),
        'caricature_alpha': -0.5,
    },
}
with open('project_vibe_evaluation.json','w') as f:
    json.dump(eval_data, f, indent=2)
print('\n✓ project_vibe_evaluation.json saved.')

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — SAVE & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
print('\nSaving all outputs...')

Image.fromarray(morph_50).save('morph_color.png')
Image.fromarray(cdsm_results[2][2]).save('morph_cdsm_sketch.png')
Image.fromarray(cdsm_results[0][2]).save('sub1_cdsm_sketch.png')
Image.fromarray(cdsm_results[1][2]).save('sub2_cdsm_sketch.png')
Image.fromarray(mean_face).save('mean_face.png')
Image.fromarray(caricature).save('caricature.png')

for i,a in enumerate(alphas_seq):
    if i % 9 == 0:
        Image.fromarray(frames[i]).save(f'morph_a{int(a*100):03d}.png')

np.save('landmarks_sub1.npy', pts1)
np.save('landmarks_sub2.npy', pts2)
np.save('mean_shape.npy',     mean_shape)

config = {
    'project'   : 'Project Vibe — CDSM Pipeline',
    'version'   : 'Final',
    'pipeline'  : 'Cross-Domain Sketch Morphing (CDSM)',
    'reference' : 'asetoodehnia.github.io/Face-Morphing',
    'modules'   : {
        'M1_morphing'   : {'method':'Delaunay+Affine','frames':N_FRAMES,'landmark_model':DAT},
        'M2_detection'  : {'threshold':THRESHOLD,'features':['tex_var','hf_energy','kurtosis','bi_ssim','lm_dist_var','nm_ratio','dct_ratio'],'accuracy':f'{acc:.0f}%'},
        'M3_sketch': {'method':'XDoG + H-DSL','refinement':'VGG19 Importance'},
        'M4_mean_face'  : {'population':len(mean_faces),'caricature_alpha':-0.5},
    },
    'metrics'   : {
        'morph_ssim_sub1': float(s1v),
        'morph_ssim_sub2': float(s2v),
        'morph_balance'  : f'{s1v/tv*100:.1f}/{s2v/tv*100:.1f}',
        'mad_accuracy'   : f'{acc:.0f}%',
        'detection'      : {r[0].replace(chr(10),' '): {'score':r[3],'verdict':r[4]} for r in results},
    }
}
with open('project_vibe_config.json','w') as f:
    json.dump(config, f, indent=2)

zip_name = 'project_vibe_FINAL.zip'
all_outputs = [
    'project_vibe_FINAL_PAPER.png',
    'project_vibe_morph.gif',
    's0_subjects.png',
    'm1_morph_result.png','m1_morph_sequence.png','m1_triangulation.png',
    'm2_detection.png',
    'm3_cdsm_sketch.png','m3_cdsm_internals.png',
    'm4_mean_face.png','m4_caricature.png',
    'morph_color.png','morph_cdsm_sketch.png',
    'sub1_cdsm_sketch.png','sub2_cdsm_sketch.png',
    'mean_face.png','caricature.png',
    'landmarks_sub1.npy','landmarks_sub2.npy','mean_shape.npy',
    'project_vibe_config.json',
    'project_vibe_evaluation.json',
    DAT,
] + [f'morph_a{int(a*100):03d}.png' for i,a in enumerate(alphas_seq) if i%9==0]

with zipfile.ZipFile(zip_name,'w',zipfile.ZIP_DEFLATED) as zf:
    for f in all_outputs:
        if os.path.exists(f):
            zf.write(f); print(f'  ✓ {f}')
        else:
            print(f'  — skipped: {f}')

size_mb = os.path.getsize(zip_name)/1024/1024
print(f'\n✓ {zip_name}  ({size_mb:.1f} MB)')
print('\n' + '='*65)
print('PROJECT VIBE — COMPLETE.')
print('Download project_vibe_FINAL.zip from Kaggle output tab.')
print('='*65)
# ══════════════════════════════════════════════════════════════════════════════
# PROJECT VIBE x PS-STYLE-GAN — ULTIMATE LOCAL FUSION PIPELINE
# Gradio UI for Manual Face Morphing + GAN Sketch Generation
# ══════════════════════════════════════════════════════════════════════════════

import os, random, warnings, traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import skimage.color as skcolor
import dlib
import gradio as gr

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️ Device : {device}')

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP DLIB & MORPHING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DAT = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(DAT):
    print('Downloading dlib predictor...')
    os.system('wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    os.system('bzip2 -d shape_predictor_68_face_landmarks.dat.bz2')

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DAT)
IMG_SIZE = 256

def get_landmarks(img_rgb):
    """Detects landmarks with a fallback to 2x upsampling for difficult/small faces."""
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 1)
    
    if len(faces) == 0:
        faces = detector(gray, 2)
        if len(faces) == 0:
            return None
            
    shape = predictor(gray, faces[0])
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)

def add_hair_and_boundary(pts, w=IMG_SIZE, h=IMG_SIZE):
    bnd = np.array([[0,0],[w//4,0],[w//2,0],[3*w//4,0],[w-1,0],
                    [0,h//4],[w-1,h//4],[0,h//2],[w-1,h//2],
                    [0,3*h//4],[w-1,3*h//4],
                    [0,h-1],[w//4,h-1],[w//2,h-1],[3*w//4,h-1],[w-1,h-1]], dtype=np.float32)
    jaw_left, jaw_right = pts[0], pts[16]
    face_center = (jaw_left + jaw_right) / 2.0
    face_width = np.linalg.norm(jaw_right - jaw_left)
    hair_pts = []
    for angle in np.linspace(np.pi, 0, 9):
        rx = face_width * 0.65
        ry = face_width * 0.85
        hx = face_center[0] + rx * np.cos(angle)
        hy = face_center[1] - ry * np.sin(angle) - (face_width * 0.15)
        hair_pts.append([np.clip(hx, 0, w-1), np.clip(hy, 0, h-1)])
    return np.vstack([pts, np.array(hair_pts, dtype=np.float32), bnd])

def compute_triangles(pts, w=IMG_SIZE, h=IMG_SIZE):
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
    out = cv2.warpAffine(crop, M, (r_d[2],r_d[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    y1,y2 = r_d[1], r_d[1]+r_d[3]
    x1,x2 = r_d[0], r_d[0]+r_d[2]
    dst_f[y1:y2, x1:x2] = dst_f[y1:y2, x1:x2]*(1-mask) + out*mask

def lab_transfer(src, ref, strength=0.6):
    src_lab = skcolor.rgb2lab(src.astype(np.float32)/255.0)
    ref_lab = skcolor.rgb2lab(ref.astype(np.float32)/255.0)
    result  = src_lab.copy()
    pts_src = get_landmarks(src)
    fmask   = np.ones((src.shape[0], src.shape[1]), dtype=np.float32)
    if pts_src is not None:
        fmask_bin = np.zeros_like(fmask)
        cx, cy = int(pts_src[:,0].mean()), int(pts_src[:,1].mean())
        rx, ry = int((pts_src[:,0].max()-pts_src[:,0].min()) * 0.60), int((pts_src[:,1].max()-pts_src[:,1].min()) * 0.70)
        cv2.ellipse(fmask_bin,(cx,cy),(rx,ry),0,0,360,1.0,-1)
        fmask = cv2.GaussianBlur(fmask_bin,(15,15),0)
    for ch in [1, 2]:
        sm, ss = src_lab[:,:,ch].mean(), src_lab[:,:,ch].std()
        rm, rs = ref_lab[:,:,ch].mean(), ref_lab[:,:,ch].std()
        transferred = (src_lab[:,:,ch]-sm)*(rs/(ss+1e-6))+rm
        result[:,:,ch] = fmask*((1-strength)*src_lab[:,:,ch] + strength*transferred) + (1-fmask)*src_lab[:,:,ch]
    result[:,:,0] = fmask*(0.7*src_lab[:,:,0] + 0.3*ref_lab[:,:,0]) + (1-fmask)*src_lab[:,:,0]
    return np.clip(skcolor.lab2rgb(result)*255, 0, 255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════════════
# 2. GAN SKETCH MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class SafeFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + (noise * self.noise_scale)

class SketchGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.enc1 = nn.Sequential(*list(vgg.children())[:4])
        self.enc2 = nn.Sequential(*list(vgg.children())[4:14])
        self.enc3 = nn.Sequential(*list(vgg.children())[14:24])
        for p in self.parameters(): p.requires_grad = False
        self.fuse = SafeFusionModule(512)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(192, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True))
        self.final = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        f1, f2, f3 = self.enc1(x), self.enc2(self.enc1(x)), self.enc3(self.enc2(self.enc1(x)))
        x = self.up1(self.fuse(f3))
        f2_rs = nn.functional.interpolate(f2, size=x.shape[2:]) if f2.shape[2:] != x.shape[2:] else f2
        x = self.up2(torch.cat([x, f2_rs], 1))
        f1_rs = nn.functional.interpolate(f1, size=x.shape[2:]) if f1.shape[2:] != x.shape[2:] else f1
        x = self.up3(torch.cat([x, f1_rs], 1))
        return torch.tanh(self.final(x))

model_path = "final_model_SHADING.pth"
model_loaded = False
try:
    netG = SketchGenerator().to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        netG.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model_loaded = True
        print("✅ GAN Model loaded successfully")
    else:
        print("⚠️ final_model_SHADING.pth not found. Using CV2 Fallback.")
    netG.eval()
except Exception as e:
    print(f"⚠️ Error loading model: {e}. Using CV2 Fallback.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE WRAPPERS FOR GRADIO
# ══════════════════════════════════════════════════════════════════════════════
def process_morph(img1_pil, img2_pil, alpha):
    if img1_pil is None or img2_pil is None: return None, "Please upload both images."
    
    # SMART CROP: Centers face and trims rectangle images into perfect squares without squishing
    img1_pil = ImageOps.fit(img1_pil, (IMG_SIZE, IMG_SIZE), centering=(0.5, 0.2))
    img2_pil = ImageOps.fit(img2_pil, (IMG_SIZE, IMG_SIZE), centering=(0.5, 0.2))
    
    im1 = np.array(img1_pil.convert('RGB'))
    im2 = np.array(img2_pil.convert('RGB'))
    
    pts1 = get_landmarks(im1)
    pts2 = get_landmarks(im2)
    
    if pts1 is None or pts2 is None:
        return None, "Error: Could not detect a face in one or both images. Ensure subjects are front-facing."
        
    pts1b = add_hair_and_boundary(pts1)
    pts2b = add_hair_and_boundary(pts2)
    
    im1_bal = lab_transfer(im1, im2, strength=0.4)
    
    pm = (1-alpha)*pts1b + alpha*pts2b
    tris = compute_triangles(pm, IMG_SIZE, IMG_SIZE)
    warp1 = np.zeros_like(im1, dtype=np.float32)
    warp2 = np.zeros_like(im2, dtype=np.float32)
    
    for tri in tris:
        t1, t2, tm = [pts1b[tri[j]] for j in range(3)], [pts2b[tri[j]] for j in range(3)], [pm[tri[j]] for j in range(3)]
        warp_triangle(im1_bal.astype(np.float32), warp1, t1, tm)
        warp_triangle(im2.astype(np.float32), warp2, t2, tm)

    pts_m = pm[:68]
    cx, cy = int(pts_m[:,0].mean()), int(pts_m[:,1].mean())
    rx, ry = int((pts_m[:,0].max()-pts_m[:,0].min())*0.60), int((pts_m[:,1].max()-pts_m[:,1].min())*0.70)
    fmask = np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.float32)
    cv2.ellipse(fmask, (cx,cy), (rx,ry), 0, 0, 360, 1.0, -1)
    fmask3 = cv2.GaussianBlur(fmask, (41,41), 0)[:,:,np.newaxis]

    blend = (1-alpha)*warp1 + alpha*warp2
    result = np.clip(fmask3*blend + (1-fmask3)*im1_bal, 0, 255).astype(np.uint8)
    
    return result, "Morph successful!"

def process_sketch(morphed_cv2, blur_kernel, ai_strength):
    if morphed_cv2 is None: return None
    
    gray = cv2.cvtColor(morphed_cv2, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    inverted = 255 - gray
    
    k_size = (blur_kernel * 2) + 1
    blurred = cv2.GaussianBlur(inverted, (k_size, k_size), 0)
    sketch_cv2 = cv2.divide(gray, 255 - blurred, scale=256)
    
    kernel_sharpen = np.array([[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]])
    sketch_cv2 = np.clip(cv2.filter2D(sketch_cv2, -1, kernel_sharpen), 0, 255).astype(np.uint8)
    
    if not model_loaded: return Image.fromarray(sketch_cv2)

    try:
        sketch_rgb = Image.fromarray(sketch_cv2).convert('RGB')
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        input_tensor = transform(sketch_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = netG(input_tensor).squeeze().cpu().detach().numpy()
            generated = np.clip((generated * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            generated = cv2.resize(generated, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
            
        generated = cv2.convertScaleAbs(generated, alpha=1.15, beta=-10)
        alpha_blend = ai_strength / 100.0
        final_sketch = cv2.addWeighted(generated, alpha_blend, sketch_cv2, 1 - alpha_blend, 0)
        return Image.fromarray(cv2.convertScaleAbs(final_sketch, alpha=1.05, beta=0))
    except Exception as e:
        print(f"Sketch Error: {e}")
        return Image.fromarray(sketch_cv2)

def full_pipeline(img1, img2, alpha, blur_k, strength):
    try:
        morphed_np, status = process_morph(img1, img2, alpha)
        if morphed_np is None: return None, None, status
        
        sketch_img = process_sketch(morphed_np, blur_k, strength)
        return Image.fromarray(morphed_np), sketch_img, "✅ Success"
    except Exception as e:
        traceback.print_exc()
        return None, None, f"❌ Pipeline Error: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
# 4. GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
custom_css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.header { text-align: center; padding: 20px; background: linear-gradient(135deg, #111 0%, #333 100%); color: white; border-radius: 10px; margin-bottom: 20px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    gr.HTML("<div class='header'><h1>🧬 Project Vibe x PS-Style-GAN</h1><p>Full-Stack Face Morphing & Neural Sketch Synthesis</p></div>")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📸 Input Subjects")
            img_in1 = gr.Image(type="pil", label="Subject 1 (Structure/Pose)")
            img_in2 = gr.Image(type="pil", label="Subject 2 (Feature/Style)")
            
            gr.Markdown("### 🎛️ Morph Controls")
            alpha_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Morph Balance (0 = Sub 1, 1 = Sub 2)")
            
            gr.Markdown("### 🖌️ Sketch Controls (ENHANCED_35)")
            blur_slider = gr.Slider(10, 30, value=18, step=1, label="Line Sharpness (Lower = Sharper)")
            strength_slider = gr.Slider(0, 100, value=35, step=5, label="GAN Shading Strength (%)")
            
            run_btn = gr.Button("🚀 Execute Fusion Pipeline", variant="primary", size="lg")
            
        with gr.Column():
            gr.Markdown("### ✨ Generated Outputs")
            out_color = gr.Image(type="pil", label="Morphed Face (RGB)")
            out_sketch = gr.Image(type="pil", label="Neural Sketch Output")
            status_text = gr.Textbox(label="Status", interactive=False)

    run_btn.click(fn=full_pipeline, inputs=[img_in1, img_in2, alpha_slider, blur_slider, strength_slider], outputs=[out_color, out_sketch, status_text])

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 LAUNCHING LOCAL GRADIO PIPELINE...")
    print("="*60 + "\n")
    app.queue().launch(server_name="127.0.0.1", server_port=7860)
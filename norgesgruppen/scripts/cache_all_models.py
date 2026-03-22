"""Cache predictions for all models, one at a time with progress."""
import pickle, numpy as np, time, json
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm

NC = 356
CONF_THR = 0.001
NMS_IOU = 0.7

def preprocess(img, sz):
    w, h = img.size
    scale = min(sz / w, sz / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    pad = np.full((sz, sz, 3), 114, dtype=np.uint8)
    px, py = (sz - nw) // 2, (sz - nh) // 2
    pad[py:py + nh, px:px + nw] = np.array(img_r)
    return np.transpose(pad.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...], scale, px, py

def nms(boxes, scores, thr=0.5):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep

def detect_raw(sess, input_name, img, ow, oh, sz=640):
    blob, scale, px, py = preprocess(img, sz)
    out = sess.run(None, {input_name: blob})[0]
    if out.ndim == 3:
        out = out[0].T if out.shape[1] == (4 + NC) else out[0]
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    cls_scores = out[:, 4:]
    max_sc = cls_scores.max(axis=1)
    mask = max_sc > CONF_THR
    x1 = np.clip((cx - bw / 2 - px) / scale, 0, ow)[mask]
    y1 = np.clip((cy - bh / 2 - py) / scale, 0, oh)[mask]
    x2 = np.clip((cx + bw / 2 - px) / scale, 0, ow)[mask]
    y2 = np.clip((cy + bh / 2 - py) / scale, 0, oh)[mask]
    boxes = np.stack([x1, y1, x2, y2], axis=1) if len(x1) > 0 else np.zeros((0, 4))
    scores = max_sc[mask]
    cls_ids = cls_scores[mask].argmax(axis=1) if len(x1) > 0 else np.zeros(0, dtype=int)
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids) if len(cls_ids) > 0 else []:
        m = cls_ids == cid
        keep = nms(boxes[m], scores[m], NMS_IOU)
        for k in keep:
            fb.append(boxes[m][k]); fs.append(float(scores[m][k])); fc.append(int(cid))
    return fb, fs, fc

models = {
    'fulldata_x': 'weights/yolov8x_640_fulldata_best_fp16_dynamic.onnx',
    'noflip_s42': 'weights/noflip_s42_fp16_dyn.onnx',
    'nf_11x_s42': 'weights/nf_11x_s42_fp16_dyn.onnx',
    'nf_l_s1': 'weights/nf_l_s1_fp16_dyn.onnx',
    'v8l_highaug': 'weights/yolov8l_640_highaug_fp16_dyn.onnx',
    'x_800_noflip': 'weights/x_800_noflip_fp16_dyn.onnx',
    'l_1280_noflip': 'weights/l_1280_noflip_fp16_dyn.onnx',
    'distilled_2phase': 'weights/distilled_2phase_fp16_dyn.onnx',
}

image_dir = Path('data/raw/coco_dataset/train/images')
img_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
print(f'{len(img_paths)} images, {len(models)} models', flush=True)

# Load existing cache if present (resume support)
cache_path = Path('cached_preds_all_models.pkl')
cache = {}
if cache_path.exists():
    cache = pickle.load(open(cache_path, 'rb'))
    print(f'Loaded existing cache with {len(cache)} models', flush=True)

for name, onnx_path in models.items():
    if name in cache:
        print(f'  {name}: CACHED (skipping)', flush=True)
        continue
    t0 = time.time()
    print(f'  {name}: loading...', end='', flush=True)
    sess = ort.InferenceSession(onnx_path, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    preds = {}
    for img_path in tqdm(img_paths, desc=f'  {name}', leave=True):
        image_id = int(img_path.stem.split('_')[-1])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        all_b, all_s, all_c = [], [], []
        for sz in [640, 800]:
            fb, fs, fc = detect_raw(sess, inp_name, img, w, h, sz=sz)
            all_b.extend(fb); all_s.extend(fs); all_c.extend(fc)
        preds[image_id] = {'boxes': all_b, 'scores': all_s, 'cls_ids': all_c, 'w': w, 'h': h}
    cache[name] = preds
    elapsed = time.time() - t0
    print(f' done in {elapsed:.1f}s', flush=True)
    # Save after each model (crash resilience)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

print(f'\nCached {len(cache)} models total', flush=True)

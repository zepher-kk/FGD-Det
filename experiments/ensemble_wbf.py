import os, sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/opt/data/private/dmt/MM5')

DATA = '/opt/data/private/dmt/datasets/M3FD_yolo/M3FD_yolo.yaml'
M1 = '/opt/data/private/runs/pmda_phase4/M3FD_Final_LearnTau/weights/best.pt'
M2 = '/opt/data/private/runs/pmda_phase4/M3FD_Ultimate_LearnTauDeepSE/weights/best.pt'
SAVE = '/opt/data/private/runs/cross_dataset/M3FD_Ensemble_WBF'

def run_val(model_path, name):

    from ultralytics import YOLOMM
    m = YOLOMM(model_path)
    save_dir = os.path.join(SAVE, name)
    os.makedirs(save_dir, exist_ok=True)
    r = m.val(data=DATA, device='cuda:0', save_json=True, save_dir=save_dir,
              project=SAVE, name=name, exist_ok=True, verbose=False)
    json_path = os.path.join(save_dir, 'predictions.json')
    if not os.path.exists(json_path):

        json_path = os.path.join(SAVE, name, 'predictions.json')
    return r, json_path

def load_coco_json(path):
    with open(path) as f:
        return json.load(f)

def wbf_fusion(preds1, preds2, iou_thr=0.55, score_thr=0.001):





    g1 = defaultdict(list); g2 = defaultdict(list)
    for p in preds1: g1[p['image_id']].append(p)
    for p in preds2: g2[p['image_id']].append(p)
    
    all_img_ids = set(g1.keys()) | set(g2.keys())
    fused = []
    
    for img_id in sorted(all_img_ids):
        p1 = g1.get(img_id, [])
        p2 = g2.get(img_id, [])
        all_p = p1 + p2
        
        if not all_p: continue
        

        boxes = np.array([[p['bbox'][0], p['bbox'][1], 
                          p['bbox'][0]+p['bbox'][2], p['bbox'][1]+p['bbox'][3]] 
                         for p in all_p])
        scores = np.array([p['score'] for p in all_p])
        cats = np.array([p['category_id'] for p in all_p])
        

        idx = np.argsort(-scores)
        boxes, scores, cats = boxes[idx], scores[idx], cats[idx]
        used = np.zeros(len(boxes), dtype=bool)
        
        for i in range(len(boxes)):
            if used[i]: continue

            same_cat = (cats == cats[i])

            bb_i = boxes[i]
            ious = []
            for j in range(len(boxes)):
                if not used[j] and cats[j] == cats[i]:
                    xi1, yi1, xi2, yi2 = bb_i
                    xj1, yj1, xj2, yj2 = boxes[j]
                    inter_w = max(0, min(xi2, xj2) - max(xi1, xj1))
                    inter_h = max(0, min(yi2, yj2) - max(yi1, yj1))
                    inter = inter_w * inter_h
                    area_i = (xi2-xi1)*(yi2-yi1)
                    area_j = (xj2-xj1)*(yj2-yj1)
                    union = area_i + area_j - inter
                    ious.append(inter/union if union>0 else 0)
                else:
                    ious.append(0)
            ious = np.array(ious)
            cluster = (ious > iou_thr) & same_cat & (~used)
            cluster_idx = np.where(cluster)[0]
            if len(cluster_idx) == 0:
                continue
            used[cluster_idx] = True


            c_scores = scores[cluster_idx]
            c_boxes = boxes[cluster_idx]
            if len(c_scores) == 0:
                continue
            w = c_scores / (c_scores.sum() + 1e-8)
            fused_box = (c_boxes * w[:, None]).sum(axis=0)
            fused_score = c_scores.max()
            

            fx, fy, fx2, fy2 = fused_box
            fw, fh = fx2-fx, fy2-fy
            if fw <= 0 or fh <= 0: continue
            if fused_score < score_thr: continue
            
            fused.append({
                'image_id': int(img_id),
                'category_id': int(cats[i]),
                'bbox': [float(fx), float(fy), float(fw), float(fh)],
                'score': float(fused_score)
            })
    
    return fused

def compute_coco_metrics(preds, gt_path):

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools not available, using approximate metrics")
        return None, None
    

    coco_gt = COCO(gt_path)
    

    coco_dt = coco_gt.loadRes(preds)
    

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    ap50 = coco_eval.stats[1]
    ap50_95 = coco_eval.stats[0]
    return ap50, ap50_95


def main():
    print('='*60)
    print('WBF Ensemble: LearnTau + LearnTauDeepSE')
    print('='*60)
    os.makedirs(SAVE, exist_ok=True)
    

    print('\n[1/2] Model 1: LearnTau (mAP50 champion)...')
    r1, j1 = run_val(M1, 'LearnTau')
    ap1_50, ap1_95 = r1.box.map50, r1.box.map
    
    print('\n[2/2] Model 2: LearnTauDeepSE (mAP50-95 champion)...')
    r2, j2 = run_val(M2, 'LearnTauDeepSE')
    ap2_50, ap2_95 = r2.box.map50, r2.box.map
    

    print('\n[Fusion] Loading predictions...')
    p1 = load_coco_json(j1) if os.path.exists(j1) else []
    p2 = load_coco_json(j2) if os.path.exists(j2) else []
    
    if not p1 or not p2:

        import glob
        alt1 = glob.glob(os.path.join(SAVE, '**', 'predictions.json'), recursive=True)
        print(f'Found JSONs: {alt1}')
        if len(alt1) >= 2:
            p1 = load_coco_json(alt1[0])
            p2 = load_coco_json(alt1[1])
    
    print(f'  LearnTau predictions: {len(p1)}')
    print(f'  LearnTauDeepSE predictions: {len(p2)}')
    

    print('\n[Fusion] Running WBF...')
    fused = wbf_fusion(p1, p2, iou_thr=0.55)
    print(f'  Fused predictions: {len(fused)}')
    

    fused_path = os.path.join(SAVE, 'predictions_fused.json')
    with open(fused_path, 'w') as f:
        json.dump(fused, f)
    

    print('\n[Evaluate] Computing COCO metrics...')
    gt_path = '/opt/data/private/dmt/datasets/M3FD_yolo/annotations/instances_val.json'
    if not os.path.exists(gt_path):

        import glob
        gt_candidates = glob.glob('/opt/data/private/dmt/datasets/M3FD_yolo/**/instances*.json', recursive=True)
        gt_path = gt_candidates[0] if gt_candidates else None
    
    if gt_path and os.path.exists(gt_path):
        ap_ens, ap_ens_95 = compute_coco_metrics(fused, gt_path)
    else:
        print(f'  GT not found at {gt_path}, skipping COCO eval')
        ap_ens, ap_ens_95 = None, None
    

    print(f'\n{"="*60}')
    print(f'RESULTS:')
    print(f'  LearnTau:           mAP50={ap1_50:.4f}  mAP50-95={ap1_95:.4f}')
    print(f'  LearnTauDeepSE:     mAP50={ap2_50:.4f}  mAP50-95={ap2_95:.4f}')
    if ap_ens is not None:
        print(f'  WBF Ensemble:       mAP50={ap_ens:.4f}  mAP50-95={ap_ens_95:.4f}')
        delta50 = ap_ens - max(ap1_50, ap2_50)
        delta95 = ap_ens_95 - max(ap1_95, ap2_95)
        print(f'  Δ vs best single:   mAP50 {delta50:+.4f}  mAP50-95 {delta95:+.4f}')
    else:
        print(f'  WBF Ensemble:       predictions saved to {fused_path}')
        print(f'                       (COCO eval requires pycocotools + GT JSON)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()


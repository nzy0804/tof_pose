#!/usr/bin/env python3
"""Simple infer service implementing cached interpolation + dual inference.

Usage (test):
  python scripts/infer_service.py inputs/img1.png inputs/img2.png ...

This will run infer sequentially on provided images, saving outputs to outputs/infer_test/.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np


class ModelService:
    def __init__(self):
        self.prev_frame = None  # numpy array (H,W) or (H,W,3)
        self.prev_frame_id = None

    def _decode_image(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError('cannot decode image')
        # normalize to single-channel depth-like image for processing
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray

    def interpolate_depth(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # simple linear interpolation and clip
        a_f = a.astype(np.float32)
        b_f = b.astype(np.float32)
        mid = ((a_f + b_f) * 0.5).astype(np.uint8)
        return mid

    def process_single(self, depth_gray: np.ndarray) -> tuple[bytes, bytes]:
        h, w = depth_gray.shape[:2]
        out_size = (320, 320)
        depth_up = cv2.resize(depth_gray, out_size, interpolation=cv2.INTER_LINEAR)

        pseudo_color = cv2.applyColorMap(depth_up, cv2.COLORMAP_MAGMA)

        skeleton_contour = np.zeros_like(pseudo_color)
        # simple mock keypoints relative to image
        kp = [
            (int(out_size[0]*0.5), int(out_size[1]*0.2)),
            (int(out_size[0]*0.5), int(out_size[1]*0.4)),
            (int(out_size[0]*0.4), int(out_size[1]*0.6)),
            (int(out_size[0]*0.6), int(out_size[1]*0.6)),
            (int(out_size[0]*0.45), int(out_size[1]*0.85)),
            (int(out_size[0]*0.55), int(out_size[1]*0.85)),
        ]
        for p in kp:
            cv2.circle(skeleton_contour, p, 6, (0, 255, 0), -1)
        cv2.line(skeleton_contour, kp[0], kp[1], (0, 255, 0), 2)
        cv2.line(skeleton_contour, kp[1], kp[2], (0, 255, 0), 2)
        cv2.line(skeleton_contour, kp[1], kp[3], (0, 255, 0), 2)
        cv2.line(skeleton_contour, kp[2], kp[4], (0, 255, 0), 2)
        cv2.line(skeleton_contour, kp[3], kp[5], (0, 255, 0), 2)

        _, thr = cv2.threshold(depth_up, 10, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best = max(cnts, key=cv2.contourArea)
            cv2.drawContours(skeleton_contour, [best], -1, (255, 255, 255), 2)

        _, b1 = cv2.imencode('.png', pseudo_color, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        _, b2 = cv2.imencode('.png', skeleton_contour, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return (b1.tobytes(), b2.tobytes())

    def infer(self, frame_id: str, image_bytes: bytes) -> dict:
        start = time.time()
        depth = self._decode_image(image_bytes)

        # first frame
        if self.prev_frame is None:
            pseudo_color, skeleton_contour = self.process_single(depth)
            self.prev_frame = depth.copy()
            self.prev_frame_id = frame_id
            elapsed = int((time.time() - start) * 1000)
            return {
                'frame_id': frame_id,
                'pseudo_color_image_s1': b'',
                'skeleton_contour_image_s1': b'',
                'pseudo_color_image_s2': pseudo_color,
                'skeleton_contour_image_s2': skeleton_contour,
                'person_count': 0,
                'processing_time_ms': elapsed,
            }

        # interpolate
        mid = self.interpolate_depth(self.prev_frame, depth)

        # process mid and current
        pseudo_color_s1, skeleton_contour_s1 = self.process_single(mid)
        pseudo_color_s2, skeleton_contour_s2 = self.process_single(depth)

        # update cache
        self.prev_frame = depth.copy()
        self.prev_frame_id = frame_id

        elapsed = int((time.time() - start) * 1000)
        return {
            'frame_id': frame_id,
            'pseudo_color_image_s1': pseudo_color_s1,
            'skeleton_contour_image_s1': skeleton_contour_s1,
            'pseudo_color_image_s2': pseudo_color_s2,
            'skeleton_contour_image_s2': skeleton_contour_s2,
            'person_count': 0,
            'processing_time_ms': elapsed,
        }


def main(argv):
    svc = ModelService()
    outdir = Path('outputs/infer_test')
    outdir.mkdir(parents=True, exist_ok=True)

    for idx, p in enumerate(argv, start=1):
        path = Path(p)
        if not path.exists():
            print('skip missing', p)
            continue
        data = path.read_bytes()
        frame_id = f'frame_{idx:06d}'
        res = svc.infer(frame_id, data)
        # save outputs
        outputs = {
            'S1_pseudo_color': res.get('pseudo_color_image_s1'),
            'S1_skeleton_contour': res.get('skeleton_contour_image_s1'),
            'S2_pseudo_color': res.get('pseudo_color_image_s2'),
            'S2_skeleton_contour': res.get('skeleton_contour_image_s2'),
        }
        for name, b in outputs.items():
            outp = outdir / f'{frame_id}_{name}.png'
            if b:
                outp.write_bytes(b)
                print(f'WROTE {outp} {len(b)} bytes')
            else:
                print(f'NO DATA for {outp}')
        print('frame', frame_id, 'proc_ms', res['processing_time_ms'])


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python scripts/infer_service.py <image1> <image2> ...')
        sys.exit(1)
    main(sys.argv[1:])

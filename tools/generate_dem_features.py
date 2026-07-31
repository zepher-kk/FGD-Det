#!/usr/bin/env python3





















import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm






class DEMKernels:


    @staticmethod
    def gaussian_kernel_2d(ksize: int, sigma: float = 0.0) -> torch.Tensor:

        if sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

        center = ksize // 2
        x = torch.arange(ksize, dtype=torch.float32) - center
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d

    @staticmethod
    def sobel_kernel_x(ksize: int = 3) -> torch.Tensor:

        if ksize == 3:
            return torch.tensor([
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0]
            ], dtype=torch.float32)
        elif ksize == 5:
            return torch.tensor([
                [-1.0, -2.0, 0.0, 2.0, 1.0],
                [-4.0, -8.0, 0.0, 8.0, 4.0],
                [-6.0, -12.0, 0.0, 12.0, 6.0],
                [-4.0, -8.0, 0.0, 8.0, 4.0],
                [-1.0, -2.0, 0.0, 2.0, 1.0]
            ], dtype=torch.float32)
        raise ValueError(f"Sobel kernel size must be 3 or 5, got {ksize}")

    @staticmethod
    def sobel_kernel_y(ksize: int = 3) -> torch.Tensor:

        return DEMKernels.sobel_kernel_x(ksize).t()

    @staticmethod
    def scharr_kernel_x() -> torch.Tensor:

        return torch.tensor([
            [-3.0, 0.0, 3.0],
            [-10.0, 0.0, 10.0],
            [-3.0, 0.0, 3.0]
        ], dtype=torch.float32)

    @staticmethod
    def scharr_kernel_y() -> torch.Tensor:

        return DEMKernels.scharr_kernel_x().t()

    @staticmethod
    def laplacian_kernel(ksize: int = 3) -> torch.Tensor:

        if ksize == 1:
            return torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0]
            ], dtype=torch.float32)
        elif ksize == 3:
            return torch.tensor([
                [2.0, 0.0, 2.0],
                [0.0, -8.0, 0.0],
                [2.0, 0.0, 2.0]
            ], dtype=torch.float32)
        elif ksize == 5:
            return torch.tensor([
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [1.0, 2.0, -16.0, 2.0, 1.0],
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0]
            ], dtype=torch.float32)
        raise ValueError(f"Laplacian kernel size must be 1, 3, or 5, got {ksize}")

    @staticmethod
    def mean_kernel(ksize: int) -> torch.Tensor:

        kernel = torch.ones(ksize, ksize, dtype=torch.float32)
        return kernel / (ksize * ksize)






class DEMFeatureGenerator(nn.Module):






    def __init__(
        self,
        gaussian_ksize: int = 3,
        sobel_ksize: int = 3,
        roughness_ksize: int = 5,
        local_diff_ksize: int = 15,
        use_scharr: bool = False,
        device: Optional[str] = None,
    ):











        super().__init__()

        self.gaussian_ksize = gaussian_ksize
        self.sobel_ksize = sobel_ksize
        self.roughness_ksize = roughness_ksize
        self.local_diff_ksize = local_diff_ksize
        self.use_scharr = use_scharr


        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)


        self._register_kernels()
        self.to(self.device)

    def _register_kernels(self):


        gaussian = DEMKernels.gaussian_kernel_2d(self.gaussian_ksize)
        self.register_buffer(
            "gaussian_kernel",
            gaussian.view(1, 1, self.gaussian_ksize, self.gaussian_ksize)
        )


        if self.use_scharr:
            grad_x = DEMKernels.scharr_kernel_x()
            grad_y = DEMKernels.scharr_kernel_y()
            self.grad_ksize = 3
        else:
            grad_x = DEMKernels.sobel_kernel_x(self.sobel_ksize)
            grad_y = DEMKernels.sobel_kernel_y(self.sobel_ksize)
            self.grad_ksize = self.sobel_ksize

        self.register_buffer("sobel_x", grad_x.view(1, 1, self.grad_ksize, self.grad_ksize))
        self.register_buffer("sobel_y", grad_y.view(1, 1, self.grad_ksize, self.grad_ksize))


        laplacian = DEMKernels.laplacian_kernel(self.sobel_ksize)
        lap_ksize = laplacian.shape[0]
        self.register_buffer("laplacian_kernel", laplacian.view(1, 1, lap_ksize, lap_ksize))


        mean_rough = DEMKernels.mean_kernel(self.roughness_ksize)
        self.register_buffer(
            "mean_kernel_rough",
            mean_rough.view(1, 1, self.roughness_ksize, self.roughness_ksize)
        )

        mean_local = DEMKernels.mean_kernel(self.local_diff_ksize)
        self.register_buffer(
            "mean_kernel_local",
            mean_local.view(1, 1, self.local_diff_ksize, self.local_diff_ksize)
        )

    def _conv2d_reflect(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:

        ksize = kernel.shape[-1]
        pad = ksize // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_padded, kernel)

    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[1] == 1:
            return x
        elif x.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
            return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        return x.mean(dim=1, keepdim=True)

    def _normalize_channel(self, x: torch.Tensor) -> torch.Tensor:

        B = x.shape[0]
        x_flat = x.view(B, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        denom = torch.clamp(max_val - min_val, min=1e-8)
        return (x - min_val) / denom

    @torch.no_grad()
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:










        if x.dtype != torch.float32:
            x = x.float()
        if x.max() <= 1.0:
            x = x * 255.0


        gray = self._rgb_to_gray(x)
        gray = self._conv2d_reflect(gray, self.gaussian_kernel)


        elevation = gray


        gx = self._conv2d_reflect(gray, self.sobel_x)
        gy = self._conv2d_reflect(gray, self.sobel_y)
        slope = torch.sqrt(gx**2 + gy**2)
        aspect = (torch.atan2(gy, gx) + math.pi) / (2 * math.pi)


        curvature = self._conv2d_reflect(gray, self.laplacian_kernel)


        mean_sq = self._conv2d_reflect(gray**2, self.mean_kernel_rough)
        sq_mean = self._conv2d_reflect(gray, self.mean_kernel_rough) ** 2
        roughness = torch.sqrt(torch.clamp(mean_sq - sq_mean, min=0))


        background = self._conv2d_reflect(gray, self.mean_kernel_local)
        local_diff = gray - background


        channels = [elevation, slope, aspect, curvature, roughness, local_diff]
        if normalize:
            channels = [self._normalize_channel(ch) for ch in channels]

        return torch.cat(channels, dim=1)

    def process_numpy(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:











        if len(image.shape) == 2:
            tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

        tensor = tensor.to(self.device)
        features = self.forward(tensor, normalize=normalize)
        return features.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def process_batch(
        self,
        images: List[np.ndarray],
        normalize: bool = True
    ) -> List[np.ndarray]:










        if not images:
            return []


        tensors = []
        for img in images:
            if len(img.shape) == 2:
                t = torch.from_numpy(img).float().unsqueeze(0)
            else:
                t = torch.from_numpy(img).float().permute(2, 0, 1)
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)
        features = self.forward(batch, normalize=normalize)


        features = features.permute(0, 2, 3, 1).cpu().numpy()
        return [features[i] for i in range(len(images))]






class MultiModalDatasetProcessor:


    def __init__(self, yaml_path: str):

        self.yaml_path = Path(yaml_path)
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.root_path = Path(self.config.get("path", "."))
        if not self.root_path.is_absolute():
            self.root_path = self.yaml_path.parent / self.root_path

    def get_modality_paths(self) -> Dict[str, str]:

        return self.config.get("modality", {"rgb": "images"})

    def get_splits(self) -> List[str]:

        return [s for s in ["train", "val", "test"] if self.config.get(s)]

    def get_image_paths(self, split: str, source_modality: str) -> List[Path]:

        modality_dirs = self.get_modality_paths()
        if source_modality not in modality_dirs:
            raise ValueError(f"Modality '{source_modality}' not found")

        split_path = Path(self.config.get(split, ""))
        subdir = split_path.name
        modality_dir = modality_dirs[source_modality]
        image_dir = self.root_path / modality_dir / subdir

        if not image_dir.exists():
            return []

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        paths = []
        for ext in extensions:
            paths.extend(image_dir.glob(f"*{ext}"))
            paths.extend(image_dir.glob(f"*{ext.upper()}"))
        return sorted(paths)

    def get_output_path(
        self, source_path: Path, source_modality: str, output_dir: str
    ) -> Path:

        modality_dirs = self.get_modality_paths()
        source_dir = modality_dirs[source_modality]
        rel_path = source_path.relative_to(self.root_path / source_dir)
        return self.root_path / output_dir / rel_path






def load_image(path: Path) -> Tuple[Path, Optional[np.ndarray]]:

    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return path, img
    except Exception:
        return path, None


def save_features(
    output_path: Path,
    features: np.ndarray,
    save_format: str = "npy"
) -> bool:

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if save_format == "npy":
            np.save(output_path.with_suffix(".npy"), features.astype(np.float32))
        elif save_format == "npz":
            np.savez_compressed(output_path.with_suffix(".npz"), features=features)
        elif save_format == "png":

            features_u8 = (features * 255).astype(np.uint8)
            cv2.imwrite(str(output_path.with_suffix(".png")), features_u8[:, :, :3])
            path2 = output_path.with_name(f"{output_path.stem}_ch456.png")
            cv2.imwrite(str(path2), features_u8[:, :, 3:6])
        return True
    except Exception:
        return False


def process_dataset(
    yaml_path: str,
    source_modality: str = "rgb",
    output_dir: str = "images_dem",
    splits: Optional[List[str]] = None,
    save_format: str = "npy",
    batch_size: int = 8,
    num_workers: int = 4,
    device: Optional[str] = None,
    generator_kwargs: Optional[Dict] = None,
) -> Dict[str, int]:


















    processor = MultiModalDatasetProcessor(yaml_path)
    generator_kwargs = generator_kwargs or {}
    generator = DEMFeatureGenerator(device=device, **generator_kwargs)

    available_splits = processor.get_splits()
    splits = splits or available_splits
    splits = [s for s in splits if s in available_splits]

    print(f"Device: {generator.device}")
    print(f"Dataset: {processor.root_path}")
    print(f"Source: {source_modality} -> Output: {output_dir}")
    print(f"Splits: {splits}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    print("-" * 60)

    stats = {"processed": 0, "errors": 0, "skipped": 0}

    for split in splits:
        image_paths = processor.get_image_paths(split, source_modality)
        if not image_paths:
            print(f"[{split}] No images found")
            continue

        print(f"\n[{split}] Processing {len(image_paths)} images...")


        with ThreadPoolExecutor(max_workers=num_workers) as io_executor:
            pbar = tqdm(total=len(image_paths), desc=f"  {split}")

            for batch_start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[batch_start:batch_start + batch_size]


                load_results = list(io_executor.map(load_image, batch_paths))


                valid_items = [(p, img) for p, img in load_results if img is not None]
                if not valid_items:
                    stats["errors"] += len(batch_paths)
                    pbar.update(len(batch_paths))
                    continue

                paths, images = zip(*valid_items)


                shapes = set(img.shape[:2] for img in images)
                if len(shapes) == 1:

                    try:
                        features_list = generator.process_batch(list(images))
                    except Exception as e:

                        features_list = [generator.process_numpy(img) for img in images]
                else:

                    features_list = [generator.process_numpy(img) for img in images]


                save_tasks = []
                for path, features in zip(paths, features_list):
                    out_path = processor.get_output_path(path, source_modality, output_dir)
                    save_tasks.append((out_path, features, save_format))

                save_results = list(io_executor.map(
                    lambda args: save_features(*args), save_tasks
                ))


                stats["processed"] += sum(save_results)
                stats["errors"] += len(batch_paths) - len(valid_items) + save_results.count(False)
                pbar.update(len(batch_paths))

            pbar.close()

    print("\n" + "=" * 60)
    print(f"Complete! Processed: {stats['processed']}, Errors: {stats['errors']}")
    return stats






def main():
    parser = argparse.ArgumentParser(
        description="Generate DEM-like 6-channel features (GPU accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dem_features.py --data /path/to/data.yaml --source rgb
  python generate_dem_features.py --data /path/to/data.yaml --source ir --batch-size 16
  python generate_dem_features.py --data /path/to/data.yaml --workers 8 --device cuda

Channels:
  1. Elevation    - Grayscale intensity (pseudo-height)
  2. Slope        - Gradient magnitude
  3. Aspect       - Gradient direction [0,1]
  4. Curvature    - Laplacian (2nd derivative)
  5. Roughness    - Local standard deviation
  6. LocalDiff    - High-pass filtered
        """
    )

    parser.add_argument("--data", "-d", type=str,
                       default="/home/zhizi/work/multimodel/ultralyticmm/datasets/tree/data.yaml",
                       help="Dataset YAML path")
    parser.add_argument("--source", "-s", type=str, default="rgb",
                       help="Source modality (default: rgb)")
    parser.add_argument("--output", "-o", type=str, default="images_dem",
                       help="Output directory name")
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                       help="Splits to process")
    parser.add_argument("--format", "-f", type=str, default="npy",
                       choices=["npy", "npz", "png"],
                       help="Output format")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                       help="GPU batch size")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="I/O worker threads")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cuda", "cpu"],
                       help="Device (auto-detect if not specified)")


    parser.add_argument("--gaussian-ksize", type=int, default=3)
    parser.add_argument("--sobel-ksize", type=int, default=3)
    parser.add_argument("--roughness-ksize", type=int, default=5)
    parser.add_argument("--local-diff-ksize", type=int, default=15)
    parser.add_argument("--use-scharr", action="store_true")

    args = parser.parse_args()

    generator_kwargs = {
        "gaussian_ksize": args.gaussian_ksize,
        "sobel_ksize": args.sobel_ksize,
        "roughness_ksize": args.roughness_ksize,
        "local_diff_ksize": args.local_diff_ksize,
        "use_scharr": args.use_scharr,
    }

    process_dataset(
        yaml_path=args.data,
        source_modality=args.source,
        output_dir=args.output,
        splits=args.splits,
        save_format=args.format,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=args.device,
        generator_kwargs=generator_kwargs,
    )


if __name__ == "__main__":
    main()


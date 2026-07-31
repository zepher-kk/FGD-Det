import argparse

from ultralytics import YOLOMM


def parse_offsets(s: str):

    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("offsets must be 'start,end,step'")
    start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
    return [(v, v) for v in range(start, end + 1, step)]


def main():
    parser = argparse.ArgumentParser(
        description="TRE — Triple-Reference Evaluation for multi-modal detection"
    )
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--weights", required=True, help="Model weights .pt path")
    parser.add_argument(
        "--offsets", type=parse_offsets, default="0,15,3",
        help="Offset range: start,end,step (default: 0,15,3)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="Device (e.g. 0, cpu)")
    parser.add_argument("--name", type=str, default="val-tre", help="Save directory name")

    args = parser.parse_args()

    print(f"Loading model from {args.weights} ...")
    model = YOLOMM(args.weights)

    print(f"Running TRE on {args.data} with offsets {args.offsets} ...")
    results = model.val_tre(
        data=args.data,
        offsets=args.offsets,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        name=args.name,
    )

    print(f"\nTRE complete. RFS_total (mAP50): {results['RFS_total_mAP50']:.4f}")
    print(f"RFS_total (mAP50-95): {results['RFS_total_mAP50_95']:.4f}")


if __name__ == "__main__":
    main()


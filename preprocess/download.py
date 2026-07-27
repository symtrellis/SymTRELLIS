import argparse
from datetime import datetime

from .dataset import data_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset raw files")
    parser.add_argument("dataset", choices=sorted(data_dict))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if args.world_size <= 0:
        parser.error("--world-size must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("--rank must be in [0, --world-size)")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")

    workspace = data_dict[args.dataset](args.dataset_dir)
    metadata = workspace.read_metadata()
    metadata["sha256"] = metadata["sha256"].astype(str)

    if "raw" not in metadata.columns:
        metadata["raw"] = False
    if not args.recompute_finished:
        metadata = metadata.loc[~metadata["raw"].eq(True)]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    metadata = metadata.iloc[start:end]

    records = workspace.download(metadata=metadata, num_workers=args.num_workers)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/download_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)


if __name__ == "__main__":
    main()

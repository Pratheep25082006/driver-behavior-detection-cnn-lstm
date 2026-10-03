from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent / "Frames_Clean_224"
CLASSES = ["Distraction", "Drinking", "Normal", "Phone", "Sleeping"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not DATA_DIR.is_dir():
        print(f"ERROR: '{DATA_DIR}' not found. Edit DATA_DIR at the top of this script.")
        return

    subjects = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()], key=lambda x: x.name.lower())

    class_totals = defaultdict(int)
    subject_class_totals = defaultdict(lambda: defaultdict(int))
    angle_totals = defaultdict(int)

    print("=" * 70)
    print("PER SUBJECT / CLASS BREAKDOWN")
    print("=" * 70)

    for subject_folder in subjects:
        subject = subject_folder.name
        print(f"\n{subject}")

        for cls in CLASSES:
            class_folder = subject_folder / cls
            if not class_folder.is_dir():
                print(f"  {cls:<12} -> MISSING FOLDER")
                continue

            images = [f for f in class_folder.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
            n = len(images)

            class_totals[cls] += n
            subject_class_totals[subject][cls] = n

            # count per angle too (angle = part of filename before the last _framenum)
            for img in images:
                stem = img.stem
                angle, _, _ = stem.rpartition("_")
                if not angle:
                    angle = stem
                angle_totals[(subject, cls, angle)] += 1

            print(f"  {cls:<12} -> {n} images")

    print("\n" + "=" * 70)
    print("TOTAL FRAMES PER CLASS (across all subjects)")
    print("=" * 70)
    for cls in CLASSES:
        print(f"  {cls:<12} -> {class_totals[cls]} frames")

    if class_totals:
        max_class = max(class_totals, key=class_totals.get)
        min_class = min(class_totals, key=class_totals.get)
        max_n, min_n = class_totals[max_class], class_totals[min_class]
        ratio = max_n / min_n if min_n > 0 else float("inf")

        print(f"\nLargest class:  {max_class} ({max_n} frames)")
        print(f"Smallest class: {min_class} ({min_n} frames)")
        print(f"Imbalance ratio: {ratio:.2f}x")

        if ratio > 1.5:
            print("\n  WARNING: noticeable class imbalance (>1.5x).")
            print("  Consider: class_weight in model.fit(), or trimming the")
            print("  larger classes, or augmenting the smaller ones more.")
        else:
            print("\n  Classes are reasonably balanced. No special handling needed.")

    print("\n" + "=" * 70)
    print("FRAME COUNTS PER SUBJECT / CLASS / ANGLE (first 30 shown)")
    print("=" * 70)
    for i, ((subject, cls, angle), n) in enumerate(sorted(angle_totals.items())):
        if i >= 30:
            print(f"  ... and {len(angle_totals) - 30} more combinations")
            break
        print(f"  {subject:<10} | {cls:<12} | angle '{angle}': {n} frames")

    print("\nDone.")


if __name__ == "__main__":
    main()
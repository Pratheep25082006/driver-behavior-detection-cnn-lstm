import cv2
from pathlib import Path

# Folder containing your videos, organized as: INPUT_FOLDER/<Person>/<Class>/<video files>
INPUT_FOLDER = Path(r"D:\DL\videos")

# Where extracted frames will be saved: OUTPUT_FOLDER/<Person>/<Class>/<frames>
OUTPUT_FOLDER = Path(__file__).resolve().parent / "Frames"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def extract_frames():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    person_folders = sorted(
        [f for f in INPUT_FOLDER.iterdir() if f.is_dir()],
        key=lambda x: x.name.lower()
    )

    for person_folder in person_folders:
        person_name = person_folder.name
        print(f"\nPerson: {person_name}")

        class_folders = sorted(
            [f for f in person_folder.iterdir() if f.is_dir()],
            key=lambda x: x.name.lower()
        )

        for class_folder in class_folders:
            class_name = class_folder.name
            output_class_folder = OUTPUT_FOLDER / person_name / class_name
            output_class_folder.mkdir(parents=True, exist_ok=True)

            videos = sorted(
                [f for f in class_folder.rglob("*")
                 if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS],
                key=lambda x: x.name.lower()
            )

            print(f"  Class: {class_name} ({len(videos)} videos)")

            for video_path in videos:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"    ERROR: could not open {video_path.name}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)

                # normalize extraction to ~5-6 fps regardless of source fps
                if fps >= 50:
                    frame_interval = 10
                elif fps >= 25:
                    frame_interval = 5
                else:
                    frame_interval = max(1, round(fps / 6)) if fps > 0 else 5

                video_name = video_path.stem
                frame_number = 0
                saved_frames = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_number % frame_interval == 0:
                        filename = f"{video_name}_{saved_frames:06d}.jpg"
                        output_path = output_class_folder / filename
                        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved_frames += 1

                    frame_number += 1

                cap.release()
                print(f"    {video_path.name}: saved {saved_frames} frames")

    print("\nDone.")


if __name__ == "__main__":
    extract_frames()
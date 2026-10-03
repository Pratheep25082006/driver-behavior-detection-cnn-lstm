from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# FOLDER SETTINGS
# ============================================================

# Input folder:
# Frames/
# ├── Person1/
# │   ├── Awake/
# │   └── Drowsy/
# ├── Person2/
# │   ├── Awake/
# │   └── Drowsy/
# └── ...

INPUT_FOLDER = Path(__file__).resolve().parent / "Frames"

# NEW OUTPUT FOLDER
OUTPUT_FOLDER = Path(__file__).resolve().parent / "Frames_Clean_224"

TARGET_SIZE = 224

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# REMOVE BLACK BORDERS
# ============================================================

def remove_black_borders(img, threshold=20, min_fraction=0.70):
    """
    Detect and remove large black borders from an image.

    threshold:
        Pixel intensity below this value is considered black.

    min_fraction:
        At least 70% of the pixels in a row/column
        must be black for it to be considered a border.
    """

    img_array = np.array(img)

    # Convert RGB image to grayscale
    gray = np.mean(img_array, axis=2)

    # --------------------------------------------------------
    # Calculate how much of each ROW is black
    # --------------------------------------------------------

    row_black_fraction = np.mean(gray < threshold, axis=1)

    # --------------------------------------------------------
    # Calculate how much of each COLUMN is black
    # --------------------------------------------------------

    col_black_fraction = np.mean(gray < threshold, axis=0)

    # Original boundaries
    top = 0
    bottom = img.height

    left = 0
    right = img.width

    # --------------------------------------------------------
    # Remove black rows from TOP
    # --------------------------------------------------------

    while (
        top < bottom
        and row_black_fraction[top] > min_fraction
    ):
        top += 1

    # --------------------------------------------------------
    # Remove black rows from BOTTOM
    # --------------------------------------------------------

    while (
        bottom > top
        and row_black_fraction[bottom - 1] > min_fraction
    ):
        bottom -= 1

    # --------------------------------------------------------
    # Remove black columns from LEFT
    # --------------------------------------------------------

    while (
        left < right
        and col_black_fraction[left] > min_fraction
    ):
        left += 1

    # --------------------------------------------------------
    # Remove black columns from RIGHT
    # --------------------------------------------------------

    while (
        right > left
        and col_black_fraction[right - 1] > min_fraction
    ):
        right -= 1

    # --------------------------------------------------------
    # Crop the image
    # --------------------------------------------------------

    cropped = img.crop(
        (left, top, right, bottom)
    )

    return cropped


# ============================================================
# RESIZE + CENTER CROP
# ============================================================

def resize_shorter_then_crop(img, target=224):
    """
    Resize the image so the shorter side becomes 224,
    then center-crop the longer side to obtain 224x224.
    """

    w, h = img.size

    # --------------------------------------------------------
    # Resize while maintaining aspect ratio
    # --------------------------------------------------------

    if w < h:

        new_w = target
        new_h = int(h * target / w)

    else:

        new_h = target
        new_w = int(w * target / h)

    img_resized = img.resize(
        (new_w, new_h),
        Image.LANCZOS
    )

    # --------------------------------------------------------
    # Calculate crop coordinates
    # --------------------------------------------------------

    left = (new_w - target) // 2
    top = (new_h - target) // 2

    right = left + target
    bottom = top + target

    # --------------------------------------------------------
    # Center crop
    # --------------------------------------------------------

    img_cropped = img_resized.crop(
        (left, top, right, bottom)
    )

    return img_cropped


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

def process_all_frames():

    # Create output folder
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find all person folders
    # --------------------------------------------------------

    person_folders = sorted(
        [
            f
            for f in INPUT_FOLDER.iterdir()
            if f.is_dir()
        ],
        key=lambda x: x.name.lower()
    )

    total_images = 0

    # ========================================================
    # PERSON LOOP
    # ========================================================

    for person_folder in person_folders:

        person_name = person_folder.name

        print()
        print("=" * 60)
        print(f"Person: {person_name}")
        print("=" * 60)

        # ----------------------------------------------------
        # Find class folders
        # ----------------------------------------------------

        class_folders = sorted(
            [
                f
                for f in person_folder.iterdir()
                if f.is_dir()
            ],
            key=lambda x: x.name.lower()
        )

        # ====================================================
        # CLASS LOOP
        # ====================================================

        for class_folder in class_folders:

            class_name = class_folder.name

            # Output:
            # Frames_Clean_224/Person/Class/

            output_class_folder = (
                OUTPUT_FOLDER
                / person_name
                / class_name
            )

            output_class_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Find images
            # ------------------------------------------------

            images = sorted(
                [
                    f
                    for f in class_folder.iterdir()
                    if (
                        f.is_file()
                        and f.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                ]
            )

            count = 0

            # =================================================
            # IMAGE LOOP
            # =================================================

            for img_path in images:

                try:

                    # -----------------------------------------
                    # Open image
                    # -----------------------------------------

                    img = Image.open(
                        img_path
                    ).convert("RGB")

                    original_size = img.size

                    # -----------------------------------------
                    # STEP 1:
                    # Remove black borders
                    # -----------------------------------------

                    img_clean = remove_black_borders(
                        img,
                        threshold=20,
                        min_fraction=0.70
                    )

                    # -----------------------------------------
                    # STEP 2:
                    # Resize and crop to 224x224
                    # -----------------------------------------

                    final_img = resize_shorter_then_crop(
                        img_clean,
                        TARGET_SIZE
                    )

                    # -----------------------------------------
                    # Check final size
                    # -----------------------------------------

                    if final_img.size != (
                        TARGET_SIZE,
                        TARGET_SIZE
                    ):

                        print(
                            f"    WARNING: "
                            f"{img_path.name} "
                            f"has size {final_img.size}"
                        )

                    # -----------------------------------------
                    # Save image
                    # -----------------------------------------

                    out_path = (
                        output_class_folder
                        / img_path.name
                    )

                    final_img.save(
                        out_path,
                        quality=95
                    )

                    count += 1
                    total_images += 1

                except Exception as e:

                    print(
                        f"    ERROR on "
                        f"{img_path.name}: {e}"
                    )

            # ------------------------------------------------
            # Class summary
            # ------------------------------------------------

            print(
                f"  Class: {class_name} "
                f"-> {count} images processed"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Total images processed: {total_images}"
    )

    print(
        f"All images saved in:"
    )

    print(
        OUTPUT_FOLDER
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_all_frames()
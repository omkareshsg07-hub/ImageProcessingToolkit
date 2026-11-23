
import cv2
import argparse
from pathlib import Path

def process_image(image, operation, kernel=5):

    if operation == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif operation == "blur":
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0)

    elif operation == "edges":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)

    elif operation == "threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )
        return result

    elif operation == "enhance":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))

        return cv2.cvtColor(
            enhanced, cv2.COLOR_LAB2BGR
        )

def main():

    parser = argparse.ArgumentParser(
        description="Simple Image Processing Toolkit"
    )

    parser.add_argument("image", help="Input image path")

    parser.add_argument(
        "--operation",
        choices=[
            "grayscale",
            "blur",
            "edges",
            "threshold",
            "enhance"
        ],
        default="edges"
    )

    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        print("Could not load image.")
        return

    result = process_image(
        image,
        args.operation,
        args.kernel
    )

    if args.output:
        output_path = args.output
    else:
        source = Path(args.image)
        output_path = (
            source.stem + "_" + args.operation + ".png"
        )

    cv2.imwrite(output_path, result)

    print("Processing complete!")
    print("Saved to:", output_path)

if __name__ == "__main__":
    main()

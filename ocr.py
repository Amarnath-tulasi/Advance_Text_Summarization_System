from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


def extract_text_from_image(image_path):
    """
    Extract text from JPG, JPEG, and PNG images using Tesseract OCR.
    """

    try:
        # Open image
        image = Image.open(image_path)

        # Convert image to RGB
        image = image.convert("RGB")

        # Convert to grayscale
        image = image.convert("L")

        # Improve contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(2.0)

        # Sharpen image
        image = image.filter(ImageFilter.SHARPEN)

        # OCR configuration
        custom_config = r"--oem 3 --psm 6"

        # Extract text
        extracted_text = pytesseract.image_to_string(
            image,
            config=custom_config
        )

        # Remove unnecessary spaces
        extracted_text = extracted_text.strip()

        if not extracted_text:
            raise ValueError(
                "No readable text was detected in the image."
            )

        return extracted_text

    except Exception as e:
        raise RuntimeError(
            f"OCR failed: {str(e)}"
        )
if __name__ == "__main__":
    image_file = "test.png"

    text = extract_text_from_image(image_file)

    print("\n========== EXTRACTED TEXT ==========\n")
    print(text)
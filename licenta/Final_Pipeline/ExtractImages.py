import fitz  # PyMuPDF
import os
from PIL import Image
import io


def extract_images_from_pdf(pdf_path: str, output_dir: str = None, rotate: int = 0) -> list[Image.Image]:
    """
    Extract one image from each page of a PDF.
    
    Args:
        pdf_path:   Path to the PDF file
        output_dir: Directory to save the extracted images.
        rotate:     Rotation to apply to each image in degrees clockwise.
                    Must be 0, 90, 180, or 270.

    Returns:
        A list of PIL.Image.Image objects, one for each page.
    """
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be 0, 90, 180, or 270, got {rotate}")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"The PDF file was not found: {pdf_path}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    images: list[Image.Image] = []

    # Example : pdf_name = "document.pdf" -> pdf_name = "document"
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        if not image_list:
            print(f"[WARN] Page {page_index + 1} does not contain any image. Skipping.")
            continue

        # Take the first image from the page
        img_info = image_list[0]
        xref = img_info[0]

        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Rotate the image if needed
        if rotate != 0:
            pil_image = pil_image.rotate(-rotate, expand=True)

        images.append(pil_image)

        if output_dir is not None:
            ext = base_image.get("ext", "png")
            save_path = os.path.join(output_dir, f"{pdf_name}_page_{page_index + 1}.{ext}")
            pil_image.save(save_path)
            print(f"Image saved: {save_path}")

    doc.close()

    print(f"\nTotal images extracted: {len(images)} from {len(doc) if not doc.is_closed else page_index + 1} pages.")
    return images


if __name__ == "__main__":
    pdf_file = "./PDF's/Poza.pdf"
    output_folder = "./extracted_images"

    extracted = extract_images_from_pdf(pdf_file, output_folder, rotate=90)
    print(f"Extracted {len(extracted)} images.")

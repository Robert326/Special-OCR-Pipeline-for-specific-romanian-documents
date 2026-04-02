import fitz
import os
from PIL import Image

def extract_images_from_pdf(pdf_path: str, output_dir: str = None, rotate: int = 0, dpi: int = 300) -> list[Image.Image]:
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be 0, 90, 180, or 270, got {rotate}")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)
    images: list[Image.Image] = []

    # Render fiecare pagina la DPI specificat
    scale = dpi / 72  # 72 e DPI-ul default PyMuPDF
    matrix = fitz.Matrix(scale, scale)

    for page_index in range(len(doc)):
        page = doc[page_index]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        pil_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        if rotate != 0:
            pil_image = pil_image.rotate(-rotate, expand=True)

        images.append(pil_image)

        if output_dir is not None:
            save_path = os.path.join(output_dir, f"{pdf_name}_page_{page_index + 1}.png")
            pil_image.save(save_path)
            print(f"Saved: {save_path}")

    doc.close()
    print(f"\nTotal: {len(images)} pagini extrase.")
    return images


if __name__ == "__main__":
    extracted = extract_images_from_pdf("test2.pdf", "extracted_images", dpi=300)
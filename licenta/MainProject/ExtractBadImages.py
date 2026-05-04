import fitz
import os
from PIL import Image


def extract_images_from_pdf(pdf_path: str, output_dir: str = None, rotate: int = 0, dpi: int = 150) -> list[Image.Image]:
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be 0, 90, 180, or 270, got {rotate}")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    images: list[Image.Image] = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    doc = fitz.open(pdf_path)

    # Factorul de scalare bazat pe DPI (PDF-urile sunt native la 72 DPI)
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)

    for page_index in range(len(doc)):
        page = doc[page_index]

        # Randează pagina — funcționează cu orice tip de conținut PDF
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if rotate != 0:
            pil_image = pil_image.rotate(-rotate, expand=True)

        images.append(pil_image)

        if output_dir is not None:
            save_path = os.path.join(output_dir, f"{pdf_name}_page_{page_index + 1}.png")
            pil_image.save(save_path)
            print(f"Salvat: {save_path}")

    doc.close()
    print(f"\nTotal pagini procesate: {len(images)}")
    return images


if __name__ == "__main__":
    pdf_file = "./PDF's/Poza.pdf"
    output_folder = "./preprocessed_images"

    extracted = extract_images_from_pdf(pdf_file, output_folder, rotate=0, dpi=200)
    print(f"Extrase {len(extracted)} imagini.")
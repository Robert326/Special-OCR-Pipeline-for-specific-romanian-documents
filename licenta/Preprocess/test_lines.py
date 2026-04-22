import cv2
import numpy as np
import sys

# Luam imaginea
img_path = 'debug_page_2_1_deskewed.png'
if len(sys.argv) > 1:
    img_path = sys.argv[1]

gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if gray_img is None:
    print(f"Nu pot gasi imaginea {img_path}")
    sys.exit(1)

_, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = binary.shape

# 1. Erase the printed vertical line
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
clean_binary = cv2.subtract(binary, vertical_lines)

# 2. CLOSE horizontally to merge words but NOT columns!
# Morph CLOSE umple spatiile mici (dintre cuvinte, ~10-20px) 
# dar lasa intacte spatiile mari (dintre coloane, care sunt > 30px)
# Atentie: in incercarea anterioara am folosit DILATE care doar umfla totul si le incaleca.
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
closed = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel_close)

# Salvati imaginea sa vedem exact cum s-a comportat 'closing-ul'
cv2.imwrite("test_closed.png", closed)

# 3. Componente conexe
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

header_rows = []

# Centrul de incredere general
center_left = int(w * 0.35)
center_right = int(w * 0.65)

for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    
    if y > h * 0.7: continue
    
    right = x + width
    
    # UN HEADER REALA: Incepe de inainte de 35% din pagina si se termina dupa 65% a paginii.
    # Coloanele independente, chiar daca sunt inclinate, nu pot acoperi aceasta latime de 30% dintr-o bucata
    # (deoarece functia 'CLOSE' a refuzat sa uneasca stanga cu dreapta lasand gaura neclintita)
    if x < center_left and right > center_right:
        for row in range(y, y + height):
            header_rows.append(row)

header_rows = sorted(list(set(header_rows)))

# 4. Gruparea pe blocuri
blocks = []
if header_rows:
    current_block = [header_rows[0]]
    for i in range(1, len(header_rows)):
        # Allow gaps of up to 80 pixels between lines in the same header bloc
        if header_rows[i] - header_rows[i-1] <= 80:
            current_block.append(header_rows[i])
        else:
            blocks.append(current_block)
            current_block = [header_rows[i]]
    blocks.append(current_block)

print(f"Am gasit {len(blocks)} blocuri:")
for idx, b in enumerate(blocks):
    print(f"Bloc {idx}: rândul {b[0]} -> {b[-1]} (span: {b[-1]-b[0]}px, rânduri: {len(b)})")

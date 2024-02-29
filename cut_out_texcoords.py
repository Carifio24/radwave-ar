from os.path import join
from PIL import Image

# 0.1
tex_coords = [[0.45, 0.769512793678919], [0.55, 0.769512793678919], [0.55, 0.6695127936789189], [0.45, 0.6695127936789189]]

# 0.09
# tex_coords = [[0.455, 0.7645127936789189], [0.545, 0.7645127936789189], [0.545, 0.6745127936789189], [0.455, 0.6745127936789189]]
with Image.open(join("images", "milkywaybar.jpg")) as img:
    tx0 = min(t[0] for t in tex_coords)
    tx1 = max(t[0] for t in tex_coords)
    ty0 = min(t[1] for t in tex_coords)
    ty1 = max(t[1] for t in tex_coords)

    bounds = [tx0 * img.width, ty0 * img.height,
            tx1 * img.width, ty1 * img.height]
    size = (round(img.width * (tx1 - tx0)),
            round(img.height * (ty1 - ty0)))
    
    sub_img = img.transform(size, Image.Transform.EXTENT, data=bounds)
    sub_img.save("milkyway_slice.jpg")

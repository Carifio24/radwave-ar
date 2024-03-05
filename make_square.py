from os.path import splitext
from PIL import Image

# Retain the same center
def make_square(filepath):
    with Image.open(filepath) as img:
        center_x = img.width / 2
        center_y = img.height / 2
        min_dim = min(img.width, img.height)
        x0 = center_x - min_dim / 2
        x1 = center_x + min_dim / 2
        y0 = center_y - min_dim / 2
        y1 = center_y + min_dim / 2

        bounds = [x0, y0, x1, y1]
        size = (min_dim, min_dim)

        sub_img = img.transform(size, Image.Transform.EXTENT, data=bounds)
        base, ext = splitext(filepath)
        base += "_square"
        sub_img.save(base + ext)


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    make_square(filepath)

from os.path import join
from PIL import Image


def texcoords_from_fraction(fraction):
    sun_position = [8121.97336612, 0., 0.]
    galaxy_square_edge = 18_500
    shift = sun_position[0]
    galaxy_image_edge = fraction * galaxy_square_edge 
    
    galaxy_points = [
        [galaxy_image_edge, 0, galaxy_image_edge],
        [galaxy_image_edge, 0, -galaxy_image_edge],
        [-galaxy_image_edge, 0, -galaxy_image_edge],
        [-galaxy_image_edge, 0, galaxy_image_edge]
    ]
    shift_point = [shift, 0, 0]
    galaxy_points = [[c + sc for c, sc in zip(p, shift_point)] for p in galaxy_points]
    
    # This is the transformation from world space -> galaxy texture space
    # We determined that the galaxy image needs a 90 degree rotation
    # and so this affine transformation accounts for that.
    # It's easier if we do this before we scale
    texcoord = lambda x, z: [(-0.5 / galaxy_square_edge) * z + 0.5, (0.5 / galaxy_square_edge) * x + 0.5]
    return [texcoord(p[0], p[2]) for p in galaxy_points]


def cutout_for_fraction(fraction):
    tex_coords = texcoords_from_fraction(fraction)
    print(tex_coords)
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
        sub_img.save(join("images", f"milkyway_slice_{fraction}.jpg"))


if __name__ == "__main__":
    import sys
    fraction = float(sys.argv[1])
    cutout_for_fraction(fraction)

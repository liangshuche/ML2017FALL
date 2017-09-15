from PIL import Image as img
import sys
in_path=sys.argv[1]
out_path="Q2.png"

im_in=img.open(in_path)
width, height=im_in.size

im_out=img.new("RGB", (width,height))
for i in range(height):
	for j in range(width):
		_r,_g,_b=im_in.getpixel((j,i))
		im_out.putpixel((j,i), (int(_r/2), int(_g/2), int(_b/2)))

im_out.save(out_path)
from PIL import Image, ImageDraw

size = 256
grad = Image.new("RGB", (size, size))
px = grad.load()
for y in range(size):
    for x in range(size):
        t = (x + y) / (2 * size)
        px[x, y] = (int(108 + (255 - 108) * t), int(92 + (122 - 92) * t), int(231 + (162 - 231) * t))

mask = Image.new("L", (size, size), 0)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=56, fill=255)

img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
img.paste(grad, (0, 0), mask)

d = ImageDraw.Draw(img)
for i, h in enumerate([80, 130, 180]):
    x0 = 56 + i * 52
    d.rounded_rectangle((x0, 206 - h, x0 + 38, 206), radius=10, fill="white")

img.save("favicon.png")
print("favicon.png jadi")
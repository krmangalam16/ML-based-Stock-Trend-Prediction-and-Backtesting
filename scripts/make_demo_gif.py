from PIL import Image
import os

root = os.path.dirname(os.path.dirname(__file__))
img_dir = os.path.join(root, 'images')
paths = [os.path.join(img_dir, 'model_comparison.png'),
         os.path.join(img_dir, 'feature_importances.png'),
         os.path.join(img_dir, 'cumulative_returns.png')]
out_path = os.path.join(img_dir, 'demo.gif')

frames = []
for p in paths:
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    im = Image.open(p).convert('RGBA')
    # resize to width 700 if larger
    maxw = 700
    if im.width > maxw:
        ratio = maxw / im.width
        newsize = (maxw, int(im.height * ratio))
        im = im.resize(newsize, Image.LANCZOS)
    frames.append(im)

# convert to P mode for GIF
frames_p = [f.convert('P', palette=Image.ADAPTIVE) for f in frames]
frames_p[0].save(out_path, save_all=True, append_images=frames_p[1:], duration=1500, loop=0)
print('Saved', out_path)

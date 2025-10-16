import os
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

class CheckImgProp:
    # constructor
    def __init__(self, curr_dir, destination_dir, img_size, img_format):
        # build path safely using pathlib's Path where '/' means 'join'
        self.main_dir = Path(curr_dir) / destination_dir
        # parents true creates the parent dir if not present
        # exist_ok true tells not to throw error if dir already exists
        self.main_dir.mkdir(parents=True, exist_ok=True)  # safe to keep
        # allow int (square) or (w, h) tuple
        self.img_size = img_size
        # expected: 'RGB' or 'L'
        self.img_format = img_format

    # check image sizes (resize if needed)
    def check_size(self):
        self._iterate('s')

    # converting image mode (RGB or grayscale 'L')
    def change_format(self):
        self._iterate('f')

    # size and format helper function
    def _iterate(self, mode):
        for name in os.listdir(self.main_dir):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            in_path = self.main_dir / name
            try:
                with Image.open(in_path) as img0:
                    # respect EXIF orientation; then work on the transposed image
                    img = ImageOps.exif_transpose(img0)

                    if mode.lower() == 's':
                        # normalize desired size
                        target_size = (
                            (self.img_size, self.img_size)
                            if isinstance(self.img_size, int)
                            else tuple(self.img_size)
                        )
                        if img.size != target_size:
                            # 'resample=Image.LANCZOS' tells Pillow how to compute new pixel values during resizing.
                            # It uses the high-quality Lanczos interpolation filter, which blends nearby pixels to
                            # preserve detail and smoothness when reducing or enlarging an image.
                            resized_img = img.resize(target_size, resample=Image.LANCZOS)
                            # stem -> file name without extension
                            stem, ext = in_path.stem, in_path.suffix
                            out_name = f"resized_{stem}{ext}"
                            out_path = self.main_dir / out_name
                            # save with matching format from extension
                            pil_format = 'JPG' if ext.lower() in ('.jpg', '.jpeg') else 'PNG'
                            # explicitly providing format incase of failure
                            resized_img.save(out_path, format=pil_format)
                            print(f"[SIZE] Saved: {out_name}")
                    elif mode.lower() == 'f':
                        if self.img_format not in ('RGB', 'L'):
                            print("Invalid format! Only 'RGB' and 'L' allowed")
                        else:
                            if img.mode != self.img_format:
                                conv_img = img.convert(self.img_format)
                                stem, ext = in_path.stem, in_path.suffix
                                prefix = 'rgb' if self.img_format == 'RGB' else 'gray'
                                out_name = f"{prefix}_{stem}{ext}"
                                out_path = self.main_dir / out_name
                                pil_format = 'JPEG' if ext.lower() in ('.jpg', '.jpeg') else 'PNG'
                                conv_img.save(out_path, format=pil_format)
                                print(f"[FORMAT] Saved: {out_name}")
                    else:
                        print("Invalid mode. Use 's' (size) or 'f' (format)")

            except UnidentifiedImageError:
                print(f"[ERROR] Not an image or corrupted: {name}")
            except FileNotFoundError:
                print(f"[ERROR] File not found: {name}")
            except Exception as e:
                print(f"[ERROR] {name}: {e}")

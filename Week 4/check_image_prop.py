import os
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import pandas as pd

class CheckImgProp:
    # constructor
    def __init__(self, curr_dir, destination_dir, img_size, img_format, healthy_class = None):
        # build path safely using pathlib's Path where '/' means 'join'
        self.main_dir = Path(curr_dir) / destination_dir
        # parents true creates the parent dir if not present
        # exist_ok true tells not to throw error if dir already exists
        self.main_dir.mkdir(parents=True, exist_ok=True)  # safe to keep
        # allow int (square) or (w, h) tuple
        self.img_size = img_size
        # expected: 'RGB' or 'L'
        self.img_format = img_format
        # healthy class suffix
        self.healthy_class = healthy_class

    # check image sizes (resize if needed)
    def check_size(self):
        return self._iterate('s')

    # converting image mode (RGB or grayscale 'L')
    def change_format(self):
        return self._iterate('f')
        
    # checking class distribution
    def class_dist(self):
        return self._iterate('cd')

    def _is_image(self, name, exts=('.jpg', '.jpeg', '.png')):
        return name.lower().endswith(exts)

    # size and format helper function
    def _iterate(self, mode):
        """
        mode:
          'cd' -> Class distribution / file counts (no image writing)
          's'  -> Resize images to self.img_size
          'f'  -> Convert images to self.img_format ('RGB' or 'L')
        """
        mode = mode.lower()
        if mode not in ('cd', 's', 'f'):
            print("Invalid mode. Use 'cd' (class distribution), 's' (size), or 'f' (format)")
            return

        class_data = []
        for veg_name in os.listdir(self.main_dir):
            class_path = self.main_dir / veg_name
            if not class_path.is_dir():
                continue  # skip stray files in main_dir

            if mode == 'cd':
                # ---------- MODE: CLASS DISTRIBUTION / COUNTS ----------
                total_classes = 0
                healthy_classes = 0
                total_files = 0
                healthy_files = 0

                for class_name in os.listdir(class_path):
                    fname_path = class_path / class_name
                    if not fname_path.is_dir():
                        continue

                    # count classes
                    total_classes += 1
                    is_healthy_class = (
                        self.healthy_class is not None
                        and self.healthy_class in class_name.lower()
                    )                    
                    if is_healthy_class:
                        healthy_classes += 1

                    # count image files in this class folder (efficient, no opens)
                    img_count = sum(1 for f in os.listdir(fname_path) if self._is_image(f))
                    total_files += img_count
                    if is_healthy_class:
                        healthy_files += img_count

                disease_classes = total_classes - healthy_classes
                disease_files = total_files - healthy_files
                data = {
                    'veg_name': veg_name,
                    'disease_classes': disease_classes,
                    'healthy_classes': healthy_classes,
                    'disease_files': disease_files,
                    'healthy_files': healthy_files,
                    'total_files': total_files,
                    
                }
                class_data.append(data)

            elif mode in ('s', 'f'):
                # ---------- MODES 's' (resize) or 'f' (format) ----------
                for class_name in os.listdir(class_path):
                    fname_path = class_path / class_name
                    if not fname_path.is_dir():
                        continue

                    for fname in os.listdir(fname_path):
                        if not self._is_image(fname):
                            continue

                        # Always point to the actual file in its class folder
                        in_path = fname_path / fname

                        try:
                            with Image.open(in_path) as img0:
                                # respect EXIF orientation; then work on the transposed image
                                img = ImageOps.exif_transpose(img0)

                                if mode == 's':
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
                                        out_path = fname_path / out_name
                                        # save with matching format from extension
                                        pil_format = 'JPEG' if ext.lower() in ('.jpg', '.jpeg') else 'PNG'
                                        # explicitly providing format incase of failure
                                        resized_img.save(out_path, format=pil_format)
                                        print(f"[SIZE]   Saved: {out_path.relative_to(self.main_dir)}")

                                elif mode == 'f':
                                    if self.img_format not in ('RGB', 'L'):
                                        print("Invalid format! Only 'RGB' and 'L' allowed")
                                    else:
                                        if img.mode != self.img_format:
                                            conv_img = img.convert(self.img_format)
                                            stem, ext = in_path.stem, in_path.suffix
                                            prefix = 'rgb' if self.img_format == 'RGB' else 'gray'
                                            out_name = f"{prefix}_{stem}{ext}"
                                            out_path = fname_path / out_name
                                            pil_format = 'JPEG' if ext.lower() in ('.jpg', '.jpeg') else 'PNG'
                                            conv_img.save(out_path, format=pil_format)
                                            print(f"[FORMAT] Saved: {out_path.relative_to(self.main_dir)}")

                        except UnidentifiedImageError:
                            print(f"[ERROR] Not an image or corrupted: {in_path.relative_to(self.main_dir)}")
                        except FileNotFoundError:
                            print(f"[ERROR] File not found: {in_path.relative_to(self.main_dir)}")
                        except Exception as e:
                            print(f"[ERROR] {in_path.relative_to(self.main_dir)}: {e}")
        if mode == 'cd':
            return pd.DataFrame(class_data)

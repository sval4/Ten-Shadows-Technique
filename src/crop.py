import os
from PIL import Image

def crop_to_model(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.webp')): # JPGs don't support transparency
            img_path = os.path.join(input_folder, filename)
            
            with Image.open(img_path) as img:
                # Ensure image is in RGBA mode so we have the Alpha channel
                img = img.convert("RGBA")
                
                # Split the image into R, G, B, and Alpha channels
                alpha = img.split()[-1]
                
                # getbbox() on the Alpha channel finds the box containing all non-transparent pixels
                bbox = alpha.getbbox()
                
                if bbox:
                    cropped_img = img.crop(bbox)
                    cropped_img.save(os.path.join(output_folder, filename))
                    print(f"Processed: {filename}")
                else:
                    print(f"Skipped {filename}: Image is entirely transparent.")

# Usage
input_dir = "nue_frames"
output_dir = "nue_frames"
crop_to_model(input_dir, output_dir)
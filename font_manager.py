import os
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

class FontManager:
    def __init__(self):
        self.font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        self.font_path = os.path.join(self.font_dir, 'NotoSansTC-Regular.otf')
        self._ensure_font_exists()
    
    def _ensure_font_exists(self):
        """Ensure the Chinese font exists, download if not present."""
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)
        
        if not os.path.exists(self.font_path):
            print("Downloading Chinese font...")
            font_url = 'https://fonts.gstatic.com/s/notosanstc/v26/-nF7OG829Oofr2wohFbTp9iFOSsLA_ZJ1g.otf'
            try:
                response = requests.get(font_url)
                response.raise_for_status()
                with open(self.font_path, 'wb') as f:
                    f.write(response.content)
                # Verify font file
                try:
                    ImageFont.truetype(self.font_path, 16)
                    print("Font downloaded and verified successfully!")
                except Exception as e:
                    print(f"Downloaded font file is invalid: {e}")
                    if os.path.exists(self.font_path):
                        os.remove(self.font_path)
                    raise
            except Exception as e:
                print(f"Failed to download font: {e}")
                raise
    
    def get_font_path(self):
        """Get the path to the Chinese font."""
        return self.font_path
    
    def put_text_pil(self, img, text, position, font_size=32, color=(255, 255, 255)):
        """Draw text on image using Pillow for better CJK support.
        
        Args:
            img: OpenCV image (numpy array)
            text: Text to draw
            position: Tuple of (x, y) coordinates
            font_size: Font size in pixels
            color: Text color in RGB
        
        Returns:
            OpenCV image with text drawn
        """
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Create draw object and font
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.font_path, font_size)
        
        # Draw text
        draw.text(position, text, font=font, fill=color)
        
        # Convert back to OpenCV format (RGB to BGR)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
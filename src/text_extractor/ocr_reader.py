import cv2
from PIL import Image
import pytesseract
import os

class OCRReader:
    def __init__(self, tesseract_cmd=None, lang='eng'):
        """
        Initialize the OCR reader with optional custom settings.
        
        :param tesseract_cmd: Path to tesseract executable
        :param lang: Language for OCR (default: 'eng')
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            # Default path for Windows - adjust as needed
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        self.lang = lang
        
    def load_image(self, image_path):
        """Load an image from the specified path."""
        try:
            if not os.path.exists(image_path):
                print(f"Error: Image file not found at '{image_path}'")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Failed to load image - file may be corrupted or in unsupported format")
                return None
                
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def preprocess_low_contrast(self, image):
        """
        Alternative preprocessing for low contrast images.
        
        :param image: Input image
        :return: Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        return binary
    
    def optimize_for_tesseract(self, image):
        """
        Apply preprocessing optimized specifically for Tesseract.
        Based on recommendations from Tesseract documentation.
        
        :param image: Input image
        :return: Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to improve OCR accuracy (300-400 DPI is recommended for Tesseract)
        # Assuming image might be lower resolution, scale by 2x
        height, width = gray.shape
        gray = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # Remove noise
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binarize
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (Tesseract works better with black text on white background)
        # Count white and black pixels to determine background
        white_pixels = cv2.countNonZero(binary)
        total_pixels = binary.size
        black_pixels = total_pixels - white_pixels
        
        # If black pixels > white pixels, background is likely black, so invert
        if black_pixels > white_pixels:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def extract_text(self, image, preprocess_method='low_contrast'):
        """
        Extract text from the given image using specified preprocessing method.
        
        :param image: Input image
        :param preprocess_method: Method to preprocess the image ( 'low_contrast', 'tesseract_optimized')
        :return: Extracted text
        """
        if image is None:
            return None
        
        # Choose preprocessing method
        # if preprocess_method == 'black_text':
        #     preprocessed = self.preprocess_black_text(image)
        if preprocess_method == 'low_contrast':
            preprocessed = self.preprocess_low_contrast(image)
        elif preprocess_method == 'tesseract_optimized':
            preprocessed = self.optimize_for_tesseract(image)
        else:
            # Default to black text preprocessing
            preprocessed = self.preprocess_low_contrast(image)
        
        # Convert OpenCV image to PIL format for pytesseract
        pil_image = Image.fromarray(preprocessed)
        
        # Extract text with custom configuration
        # PSM modes: 6 = assume a single uniform block of text
        #           3 = fully automatic page segmentation
        # OEM modes: 1 = LSTM only
        config = '--psm 6 --oem 1'
        
        text = pytesseract.image_to_string(
            pil_image, 
            lang=self.lang,
            config=config
        )
        
        return text
    
    def try_all_methods(self, image):
        """
        Try all preprocessing methods and return results.
        
        :param image: Input image
        :return: Dictionary of results with preprocessing method as key
        """
        results = {}
        methods = [ 'low_contrast', 'tesseract_optimized']
        
        for method in methods:
            text = self.extract_text(image, method)
            results[method] = text
            
        return results
    
    def save_debug_image(self, image, filename):
        """
        Save preprocessed image for debugging purposes.
        
        :param image: Preprocessed image
        :param filename: Output filename
        """
        cv2.imwrite(filename, image)
        print(f"Debug image saved as {filename}")
    
    def process_image(self, image_path, preprocess_method='low_contrast', try_all=False, save_debug=False):
        """
        Process an image and extract text.
        
        :param image_path: Path to the image
        :param preprocess_method: Method to preprocess the image
        :param try_all: Whether to try all preprocessing methods
        :param save_debug: Whether to save preprocessed images for debugging
        :return: Extracted text or dictionary of results if try_all=True
        """
        image = self.load_image(image_path)
        if image is None:
            return None
            
        if try_all:
            results = self.try_all_methods(image)
            
            if save_debug:
                # Save debug images for each method
                for method in [ 'low_contrast', 'tesseract_optimized']:
                    # if method == 'black_text':
                    #     debug_img = self.preprocess_black_text(image)
                    if method == 'low_contrast':
                        debug_img = self.preprocess_low_contrast(image)
                    else:
                        debug_img = self.optimize_for_tesseract(image)
                    
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    debug_filename = f"{base_name}_{method}_debug.png"
                    self.save_debug_image(debug_img, debug_filename)
            
            return results
        else:
            if save_debug:
                # if preprocess_method == 'black_text':
                #     debug_img = self.preprocess_black_text(image)
                if preprocess_method == 'low_contrast':
                    debug_img = self.preprocess_low_contrast(image)
                else:
                    debug_img = self.optimize_for_tesseract(image)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                debug_filename = f"{base_name}_{preprocess_method}_debug.png"
                self.save_debug_image(debug_img, debug_filename)
            
            return self.extract_text(image, preprocess_method)
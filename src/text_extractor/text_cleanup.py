from google import genai
from constants.settings import settings
from .ocr_reader import OCRReader

class TextCleanup:
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def extractText (self) -> str:
        ocr_reader = OCRReader()
        image_path = "assets/image.png"
        results = ocr_reader.process_image(image_path,"low_contrast", try_all=True,debug_path="assets/preprocessed/" )
        print("ocr reader results",type(results))
        for method, text in results.items():
            print(f"\n--- Method: {method} ---")
            print(text if text else "[No text detected]")
        
        return results["tesseract_optimized"]
    def generate_text(self, ) -> str:
        extracted_text =  self.extractText()
        print("extracted text",extracted_text)
        full_prompt = f"{extracted_text} : Based on the two method can you cleanup the unnecessary text and give me only the question and answer in the form of a question and answer."
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt
            )
            return response.text
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
        

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from ocr_reader import OCRReader  # Import your existing OCRReader class

class TextCleanup:
    def __init__(self, tesseract_cmd=None, lang='eng', model_name="facebook/bart-large-cnn"):
        """
        Initialize the OCR processor with Hugging Face integration.
        
        :param tesseract_cmd: Path to tesseract executable
        :param lang: Language for OCR (default: 'eng')
        :param model_name: HuggingFace model for text cleaning/processing
        """
        # Initialize OCR
        self.ocr_reader = OCRReader(tesseract_cmd, lang)
        
        # Initialize Hugging Face models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load text cleaning model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Initialize NER pipeline for entity recognition
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=0 if self.device == "cuda" else -1)
        
        # Text correction pipeline
        self.text_correction = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base", device=0 if self.device == "cuda" else -1)
    
    def extract_and_process(self, image_path):
        """
        Extract text from image using multiple methods and process with HuggingFace models.
        
        :param image_path: Path to image file
        :return: Dictionary with original and processed results
        """
        # Extract text using multiple methods
        ocr_results = self.ocr_reader.process_image(image_path, try_all=True)

        print("OCR results:", ocr_results)
        
        processed_results = {
            "original": ocr_results,
            "cleaned": {},
            "entities": {},
            "corrected": {}
        }
        
        # Process each extraction result
        for method, text in ocr_results.items():
            if text:
                # Clean and summarize text
                cleaned_text = self.clean_text(text)
                processed_results["cleaned"][method] = cleaned_text
                
                # Extract entities
                entities = self.extract_entities(text)
                processed_results["entities"][method] = entities
                
                # Correct spelling and grammar
                corrected_text = self.correct_text(text)
                processed_results["corrected"][method] = corrected_text
        
        # Determine best result
        processed_results["best_result"] = self.determine_best_result(processed_results)
        
        return processed_results
    
    def clean_text(self, text):
        """
        Clean and summarize text using BART model.
        
        :param text: Input text
        :return: Cleaned/summarized text
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            max_length=150,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        cleaned_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return cleaned_text
    
    def extract_entities(self, text):
        """
        Extract named entities from text.
        
        :param text: Input text
        :return: List of extracted entities
        """
        entities = self.ner_pipeline(text)
        
        # Group entities by type
        grouped_entities = {}
        for entity in entities:
            entity_type = entity["entity"]
            entity_text = entity["word"]
            
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
                
            if entity_text not in grouped_entities[entity_type]:
                grouped_entities[entity_type].append(entity_text)
        
        return grouped_entities
    
    def correct_text(self, text):
        """
        Correct spelling and grammar in text.
        
        :param text: Input text
        :return: Corrected text
        """
        # Break text into chunks if too long
        max_chunk_size = 512
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        corrected_chunks = []
        for chunk in chunks:
            correction = self.text_correction(chunk, max_length=len(chunk) + 100)[0]["generated_text"]
            corrected_chunks.append(correction)
        
        return " ".join(corrected_chunks)
    
    def determine_best_result(self, processed_results):
        """
        Determine the best result from multiple processing methods.
        
        :param processed_results: Dictionary of processed results
        :return: Best method and its corrected text
        """
        best_method = None
        best_score = -1
        
        # Simple heuristic: method with longest corrected text and fewest newlines
        for method, text in processed_results["corrected"].items():
            if text:
                # Score based on text length minus newline count
                score = len(text) - (text.count('\n') * 5)
                
                if score > best_score:
                    best_score = score
                    best_method = method
        
        if best_method:
            return {
                "method": best_method,
                "text": processed_results["corrected"][best_method]
            }
        else:
            return None
    
    def compare_texts(self, text1, text2):
        """
        Compare two extracted texts and merge them intelligently.
        
        :param text1: First extracted text
        :param text2: Second extracted text
        :return: Merged/best text
        """
        # If one text is significantly longer than the other, prefer it
        if len(text1) > len(text2) * 1.5:
            return text1
        elif len(text2) > len(text1) * 1.5:
            return text2
        
        # Otherwise, correct both and take the one with better score
        corrected1 = self.correct_text(text1)
        corrected2 = self.correct_text(text2)
        
        # Simple scoring: longer text minus newlines
        score1 = len(corrected1) - (corrected1.count('\n') * 5)
        score2 = len(corrected2) - (corrected2.count('\n') * 5)
        
        return corrected1 if score1 >= score2 else corrected2
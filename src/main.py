from text_extractor.text_cleanup import TextCleanup
 

if __name__ == "__main__":
    text_cleanup = TextCleanup(api_key="AIzaSyA9mBH-l3nB-MTeuPkfO8_1SNFN86rpCrw")
    res = text_cleanup.generate_text()
    print("gemini response",res)
   
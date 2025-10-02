from typing import Dict


class BaseTextAnalysis:
    
    def __init__(self):
        self.texts = {}
    
    def load_text(self, filename: str, name: str) -> bool:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            self.texts[name] = text
            print(f"Loaded text '{name}': {len(text)} characters")
            return True
        except FileNotFoundError:
            print(f"File {filename} not found!")
            return False
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return False
    
    def get_all_texts(self, texts_dict: Dict[str, str]):
        for name, filename in texts_dict.items():
            self.load_text(filename, name)
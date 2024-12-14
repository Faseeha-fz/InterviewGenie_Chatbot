import pandas as pd
import spacy

# Initialize spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_text(text: str):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return tokens, entities

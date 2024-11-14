import spacy
import yake
from gensim.models import KeyedVectors
import numpy as np
import nltk
import pandas as pd
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pdfplumber
from collections import Counter
from multiprocessing import Pool
import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cosine
nlp = spacy.load("en_core_web_sm")
import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import os



def load_word2vec_model(file_path):
    print("Loading Word2Vec model from .bin file...")
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    print("Model loaded.")
    return model


def read_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
            return text
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_keywords(text, percentage=0.2):
    word_count = len(text.split())
    num_keywords = int(word_count * percentage)
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [word for word, score in keywords]

def extract_subject(text):
    doc = nlp(text)

    for token in doc:
        if token.dep_ == "nsubj" and token.text != "’s" and token.dep_ != "poss" and not any(child.text == "’s" for child in token.children):
            return token.text

  
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"] and "’s" not in ent.text:
            return ent.text

   
    for token in doc:
        if token.pos_ == "PRON" and token.dep_ != "poss" and token.text != "’s":
            return token.text
    for token in doc:
        if token.pos_ == "NOUN" and token.dep_ != "poss" and token.text != "’s":
            return token.text

    return None

word2vec_model = load_word2vec_model('/content/unzippedGender/GoogleNews-vectors-negative300.bin')


female_words = ["she", "her","hers", "woman", "female", "sister", "mother", "daughter", "girl", "wife" ,"feminine", "lady", "aunt", "grandmother"]
male_words = ["he", "him", "his", "man", "male", "brother", "father", "son", "boy", "masculine", "husband", "gentleman", "uncle", "grandfather"]


def calculate_average_vector(word_list, model):
    vectors = []
    for word in word_list:
        if word in model:
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None

female_vector = calculate_average_vector(female_words, word2vec_model)
male_vector = calculate_average_vector(male_words, word2vec_model)


def expand_gendered_terms(seed_vector, model, similarity_threshold=0.7, max_terms=10):
    similar_words = model.most_similar(positive=[seed_vector], topn=50)
    selected_terms = []
    for word, similarity in similar_words:
        if similarity >= similarity_threshold and len(selected_terms) < max_terms:
            selected_terms.append(word)
    return selected_terms

def classify_text_by_pronouns(text):
    doc = nlp(text)

   
    female_pronouns = {"she", "her", "hers", "herself"}
    male_pronouns = {"he", "him", "his", "himself"}

    
    female_count = 0
    male_count = 0
    neutral_count = 0  

   
    for token in doc:
        if token.pos_ == "PRON":
            pronoun = token.text.lower()
            if pronoun in female_pronouns:
                print(f"\nIdentified Subject: '{pronoun}'")
                female_count += 1
            elif pronoun in male_pronouns:
                print(f"\nIdentified Subject: '{pronoun}'")
                male_count += 1
            else:
                print(f"\nIdentified Subject: '{pronoun}'")
                neutral_count += 1
        elif token.text.lower() in female_words:
            female_count += 1
        elif token.text.lower() in male_words:
            male_count += 1

    
    if female_count > 0 and male_count == 0:
        return "female-inclined"
    elif male_count > 0 and female_count == 0:
        return "male-inclined"
    elif female_count > male_count:
        return "female-inclined"
    elif male_count > female_count:
        return "male-inclined"
    elif female_count == 0 and male_count == 0:
        return "neutral"  

    return "neutral" 


def gender_inclination(word, model, female_vector, male_vector, threshold=0.03):

    if word in female_words:
        return "female-inclined"
    elif word in male_words:
          return "male-inclined"

    if word not in model:
        return "neutral"

    word_vector = model[word]

   
    female_similarity = 1 - cosine(word_vector, female_vector)
    male_similarity = 1 - cosine(word_vector, male_vector)

    
    if abs(female_similarity - male_similarity) < threshold:
        return "neutral"
    elif female_similarity > male_similarity:
        return "female-inclined"
    else:
        return "male-inclined"


dynamic_male_terms = expand_gendered_terms(male_vector, word2vec_model)
dynamic_female_terms = expand_gendered_terms(female_vector, word2vec_model)

print("Dynamically identified male terms:", dynamic_male_terms)
print("Dynamically identified female terms:", dynamic_female_terms)


def text_gender_inclination(text, model, female_vector, male_vector):
    keywords = extract_keywords(text)
    subject = extract_subject(text)
    female_count, male_count, neutral_count = 0, 0, 0

 
    print("Important Words and their Gender Inclinations:")
    for word in keywords:
        inclination = gender_inclination(word, model, female_vector, male_vector)
        print(f"'{word}': {inclination}")

  
        if inclination == "female-inclined":
            female_count += 1
        elif inclination == "male-inclined":
            male_count += 1
        else:
            neutral_count += 1


    subject_inclination = "neutral"
    if subject:
        if subject.upper() not in {"I", "YOUR", "THEIRS"}:
            subject_inclination = classify_text_by_pronouns(text)
            print(f"\nIdentified Subject: '{subject}' ({subject_inclination})")

       
            if subject_inclination == "female-inclined":
                female_count += 1
            elif subject_inclination == "male-inclined":
                male_count += 1


    if female_count == 0 and male_count == 0:
        return "neutral text"
    elif subject_inclination == "female-inclined":
        return "female-inclined text"  
    elif subject_inclination == "male-inclined":
        return "male-inclined text"
    elif female_count > male_count:
        return "female-inclined text"
    elif male_count > female_count:
        return "male-inclined text"
    else:
        return "neutral text"


text = read_text_file('/content/1.txt')

text_inclination = text_gender_inclination(text, word2vec_model, female_vector, male_vector)
print(f"\nThe text is classified as: {text_inclination}")


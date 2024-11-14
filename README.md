# Detecting-Gender-Bias-in-Software-Engineering-Education
Detecting Gender Bias in Software Engineering Education: A Keyword and Word Vector Analysis Approach
This repository hosts the code and resources used for the study: "Detecting Gender Bias in Software Engineering Education: A Keyword and Word Vector Analysis Approach". The project applies keyword extraction and word vector analysis to identify gender associations within software engineering educational content, aiming to foster inclusivity in technical education.

Overview
This project explores the presence of gender bias in educational resources by:

Extracting keywords from selected texts using the YAKE (Yet Another Keyword Extractor) algorithm.
Mapping keywords to vector representations using the Google News pretrained word embedding model.
Assigning gender labels (male, female, or neutral) to keywords based on their similarity to predefined gendered word lists.
Aggregating the counts of each gender label to determine the overall gender orientation of the text.
Our goal is to provide an objective, scalable approach for analyzing and mitigating gender biases in software engineering educational materials.

Methodology
The approach consists of five main steps:

Data Collection: Collect educational materials, focusing on widely used textbooks in software engineering.
Keyword Extraction: Use the YAKE algorithm to identify contextually significant keywords.
Word Vector Representation: Transform keywords into vectors using the Google News model to capture semantic relationships.
Gender Classification: Classify each keyword based on cosine similarity with male and female word lists.
Overall Gender Orientation: Aggregate classified keywords to identify the text's gender orientation.
Key Components
YAKE for Keyword Extraction: An unsupervised model independent of external databases, used here to extract the most relevant keywords.
Google News Word Embeddings: A pretrained model for vectorizing keywords and calculating similarity scores.
Gender Classification Logic: Cosine similarity with a threshold of 0.03 to categorize keywords

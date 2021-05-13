# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:09:27 2021

@author: jacmckenna
"""
import spacy
Example_Sentence = "Patients who in late middle age have smoked 20 cigarettes a day since their teens constitute an at-risk group. One thing they’re clearly at risk for is the acute sense of guilt that a clinician can incite, which immediately makes a consultation tense."
nlp = spacy.load('en_core_web_sm')
doc = nlp(Example_Sentence)

def spacy_process(text):
    doc = nlp(text)
    
#Tokenization and lemmatization are done with the spacy nlp pipeline commands
    lemma_list = []
    for token in doc:
        lemma_list.append(token.lemma_)
    print("Tokenize+Lemmatize:")
    print(lemma_list)
    
    #Filter the stopword
    filtered_sentence =[] 
    for word in lemma_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    
    #Remove punctuation
    punctuations="?:!.,;"
    for word in filtered_sentence:
        if word in punctuations:
            filtered_sentence.remove(word)
    print(" ")
    print("Remove stopword & punctuation: ")
    print(filtered_sentence)
    
spacy_process(Example_Sentence)
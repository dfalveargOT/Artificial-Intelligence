"""
AI_Applications : Natural Language Processing
David Felipe Alvear Goyes
Artificial Intelligence Course 2020 - I
Columbia University
05/2020
"""

import os
import re
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class NLP(object):
    def __init__(self, train_dataset=[], test_dataset=[], stop_words=[], verbose=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        with open("stopwords.en.txt",'r') as words_file:
            Word_List = words_file.read().splitlines()
        self.stop_words = set(Word_List)
        self.verbose = verbose
    
    def unigram_model(self, mode = ""):
        # create the transform sklearn function
        
        if mode == "":
            vectorizer = CountVectorizer(ngram_range=(1,1), stop_words=self.stop_words)
        elif mode == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range=(1,1), stop_words=self.stop_words)

        ## Learn the vocabulary with the train data
        vectorizer.fit(self.train_dataset["text"]) 
        matrix = vectorizer.transform(self.train_dataset["text"]) # take the "text" and encode with the vocabulary
        #encoded = matrix.toarray()

        ## compress array using sparsity matrix
        #sparse = 1.0 - np.count_nonzero(encoded)/encoded.size
        encoded = csr_matrix(matrix)

        ## Create the SGD model and train it 
        classifier = SGDClassifier(loss="hinge", penalty="l1", verbose=self.verbose, n_jobs=-1)#, max_iter=200)
        #print("train")
        classifier.fit(encoded, self.train_dataset["polarity"])

        ## Encode the test data
        #print("test")
        matrix = vectorizer.transform(self.test_dataset["text"]) # take the "text" and encode with the vocabulary
        #encoded = matrix.toarray()

        ## compress array using sparsity matrix
        encoded = csr_matrix(matrix)
        
        ## classify the test data
        outcome = classifier.predict(encoded)

        data_manager.write_outcomes(outcome, "unigram", mode)

    
    def bigram_model(self, mode = ""):
        # create the transform sklearn function
        
        if mode == "":
            vectorizer = CountVectorizer(ngram_range=(1,2), stop_words=self.stop_words)
        elif mode == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words=self.stop_words)

        ## Learn the vocabulary with the train data
        vectorizer.fit(self.train_dataset["text"]) 
        matrix = vectorizer.transform(self.train_dataset["text"]) # take the "text" and encode with the vocabulary
        #encoded = matrix.toarray()

        ## compress array using sparsity matrix
        encoded = csr_matrix(matrix)

        ## Create the SGD model and train it 
        classifier = SGDClassifier(loss="hinge", penalty="l1", verbose=self.verbose, n_jobs=-1)#, max_iter=200)
        #print("train")
        classifier.fit(encoded, self.train_dataset["polarity"])

        ## Encode the test data
        #print("test")
        matrix = vectorizer.transform(self.test_dataset["text"]) # take the "text" and encode with the vocabulary
        #encoded = matrix.toarray()

        ## compress array using sparsity matrix
        encoded = csr_matrix(matrix)
        
        ## classify the test data
        outcome = classifier.predict(encoded)

        data_manager.write_outcomes(outcome, "bigram", mode)



class data_manager(object):
    
    def __init__(self, train_dir):

        self.stop_words = np.loadtxt("./stopwords.en.txt", dtype=np.str)
        self.chars = ".,?!;:-_/<>'~`@´#$%^&*()+=\[\]\{\}|\\\"\'"

    def text_processing(self, text):
        #text = pd.read_csv(name, encoding="ISO-8859-1")
        # Remove characteres and punctuation
        text = text.replace('<br />', '')
        for char_value in self.chars:
            text = text.replace(char_value, " ")
        # Remove stopwords
        text = text.split()
        resultwords  = [word for word in text if word.lower() not in self.stop_words]
        text = ' '.join(resultwords)
        # Add to the group
    
        return text
    
    def read_txt(self, directory):
        files = os.listdir(directory)
        data = []

        for file in files:
            text = open(os.path.join(directory, file))
            text = str(text.read())
            # Remove characteres and punctuation
            text = text.replace('<br />', '')
            for char_value in self.chars:
                text = text.replace(char_value, " ")
            # Remove stopwords
            text = text.split()
            resultwords  = [word for word in text if word.lower() not in self.stop_words]
            text = ' '.join(resultwords)
            # Add to the group
            data.append(text)
        
        return data

    def imdb_data_preprocess(self, inpath, outpath="./", name="imdb_tr.csv", mix=False):
        '''
        Implement this module to extract
        and combine text files under train_path directory into 
        imdb_tr.csv. Each text file in train_path should be stored 
        as a row in imdb_tr.csv. And imdb_tr.csv should have two 
        columns, "text" and label
        '''

        # Paths to the raw data
        positive_dir = os.path.join(inpath, "pos")
        negative_dir = os.path.join(inpath, "neg")
        
        ## Read and process the data
        positive_data = self.read_txt(positive_dir)
        negative_data = self.read_txt(negative_dir)

        ## Create the dataframe with the preprocessed data
        pos_pd = pd.DataFrame({"text":positive_data, "polarity":[1 for i in range(len(positive_data))]})
        neg_pd = pd.DataFrame({"text":negative_data, "polarity":[0 for i in range(len(negative_data))]})
        dataset = pd.concat([pos_pd, neg_pd])

        ## Save the file in the folder
        dataset.to_csv(os.path.join(outpath, name))

    def load_datasets(self, train_path, test_path):
        train_dataset = pd.read_csv(train_path, engine='python', encoding="utf-8")
        test_dataset = pd.read_csv(test_path, engine='python', encoding="ISO-8859-1")
        test_dataset["text"] = test_dataset["text"].apply(self.text_processing)

        return train_dataset, test_dataset
    
    @staticmethod
    def write_outcomes(outcome, model_name, mode):
        name = model_name + mode + "." + "output.txt"
        with open(name,'w') as save_file:
            for val in outcome:
                save_file.write("{}\n".format(val))
        print("saved outcomes : " + name)

def main():
    #train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
    #test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

    train_path = ""
    test_path = "./imdb_te.csv"

    # Preprocessing the data
    dataManager = data_manager(train_path)
    #dataManager.imdb_data_preprocess(train_path)

    # Load the train and test dataset
    print("Loading Dataset")
    train_dataset, test_dataset = dataManager.load_datasets("./imdb_tr.csv", test_path)

    # Unigram Classification

    print("NLP Classifiers")
    NLP_classifier = NLP(train_dataset, test_dataset, stop_words=dataManager.stop_words)
    NLP_classifier.unigram_model()
    NLP_classifier.unigram_model(mode="tfidf")
    NLP_classifier.bigram_model()
    NLP_classifier.bigram_model(mode="tfidf")


if __name__ == "__main__":
    #pass
    main()
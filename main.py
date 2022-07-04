
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classifiers import *
import argparse




if __name__ == "__main__":

    arguments = argparse.ArgumentParser()
    arguments.add_argument('--classifier',type=str,help='classifier name')
    arguments.add_argument('--dataset',type=str,help='dataset name')
    arguments.add_argument('--scaling',type=str,help='for feature scaling enter 1')
    args = arguments.parse_args()
    classifier = args.classifier
    source_data = args.sourceOfData
    scaling = args.scaling

    
    dataset = args.dataset

    print ("your chosen classifier is ",classifier)

    if ( dataset[0].isdigit()):
        print("Error! The given dataset name starts with number")
    else:
        classi = classifiers(classifier, dataset ,source_data,scaling)
        classi.read_dataset()
        classi.run_classifier()
  

    

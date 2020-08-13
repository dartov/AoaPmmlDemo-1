import pandas as pd
import os
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """


    # load data & engineer
    iris_df = pd.read_csv(data_conf['location'])
    train, _ = train_test_split(iris_df, test_size=0.5, random_state=42)
    X = train.drop("species", 1)
    y = train['species']

    print("Starting training...")
    # fit model to training data
    classifier = PMMLPipeline([('classifier',RandomForestClassifier())])
    classifier.fit(X,y.values.ravel())
    print("Finished training")

    # export model artefacts to models/ folder
    if not os.path.exists('models'):
        os.makedirs('models')
    sklearn2pmml(classifier,"models/model.pmml")
    print("Saved trained model")

import pandas as pd
from sklearn import metrics
import json
from jpmml_evaluator.pyjnius import jnius_configure_classpath, PyJNIusBackend
from jpmml_evaluator import make_evaluator
from sklearn.model_selection import train_test_split


def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    predict_df = pd.read_csv(data_conf['location'])
    _, test = train_test_split(predict_df, test_size=0.5, random_state=42)
    X_predict = test.drop("species", 1)
    y_test = test['species']

    jnius_configure_classpath()
    backend = PyJNIusBackend()
    
    evaluator = make_evaluator(backend, "models/model.pmml") \
    .verify()

    y_predict = evaluator.evaluateAll(X_predict)
    
    scores = {}
    scores['accuracy'] = metrics.accuracy_score(y_test, y_predict['y'])
    print("model accuracy is ", scores['accuracy'])

    # dump results as json file evaluation.json to models/ folder
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
    print("Evaluation complete...")
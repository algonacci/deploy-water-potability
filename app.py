from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
 if request.method == 'POST':
    file = request.files['csvFile']
    if file:
        df = pd.read_csv(file)
        df.fillna(df.mean(), inplace=True)
        df = df.sample(frac=1).reset_index(drop=True)

        X = df.drop('Potability', axis=1)
        y = df['Potability']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = LazyClassifier(ignore_warnings=True, custom_metric=None, classifiers="all")
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        print(models)

        print(type(models))

        models_html = models.to_html(classes="table table-striped")

        return render_template('index.html', result=models_html, data=df.to_html(classes="table table-striped"))
 else:
    return render_template('index.html')



if __name__ == '__main__':
 app.run(debug=True)
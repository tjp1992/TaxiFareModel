# imports
from data import get_data, clean_data
from utils import compute_rmse
import encoders

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', encoders.DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
            ('time_enc', encoders.TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])


    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    print('Collecting Data')
    df = get_data()
    # clean data
    print('Cleaning Data')
    df = clean_data(df)
    # set X and y
    print('Training Model')
    y = df.pop("fare_amount")
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    # evaluate
    print('Evaluating Model')
    print(trainer.evaluate(X_test, y_test))
    # print('TODO')

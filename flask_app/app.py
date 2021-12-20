import os
import sys
import logging
import pickle
import time
import warnings
import pandas as pd
import numpy as np
from flask import Flask, request
from flask_restx import Api, Resource
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')


model2task = {'LogisticRegression': 'classification',
              'SVC': 'classification',
              'DecisionTreeClassifier': 'classification',
              'RandomForestClassifier': 'classification',
              'Ridge': 'regression',
              'SVR': 'regression',
              'DecisionTreeRegressor': 'regression',
              'RandomForestRegressor': 'regression'}

model2grid = {'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100],
                                     'solver': ['newton-cg', 'lbfgs', 'sag'],
                                     'penalty': ['l2', 'none']},
              'SVC': {'C': [0.01, 0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'degree': [3, 5, 8]},
              'DecisionTreeClassifier': {'max_depth': [5, 10, 20, None],
                                         'min_samples_split': [2, 5, 10],
                                         'min_samples_leaf': [1, 2, 5],
                                         'max_features': ['sqrt', 'log2', None]},
              'RandomForestClassifier': {'n_estimators': [100, 200, 300, 400, 500],
                                         'max_depth': [5, 10, 20, None],
                                         'min_samples_split': [2, 5, 10],
                                         'min_samples_leaf': [1, 2, 5],
                                         'max_features': ['sqrt', 'log2', None]},
              'Ridge': {'alpha': np.linspace(0, 1, 11),
                        'fit_intercept': [True, False],
                        'solver': ['svd', 'lsqr', 'sag']},
              'SVR': {'C': [0.01, 0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'degree': [3, 5, 8]},
              'DecisionTreeRegressor': {'max_depth': [5, 10, 20, None],
                                        'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 5],
                                        'max_features': ['sqrt', 'log2', None]},
              'RandomForestRegressor': {'n_estimators': [100, 200, 300, 400, 500],
                                        'max_depth': [5, 10, 20, None],
                                        'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 5],
                                        'max_features': ['sqrt', 'log2', None]}}


def get_model(model, params):
    if model == 'LogisticRegression':
        return LogisticRegression(**params)
    elif model == 'SVC':
        return SVC(**params)
    elif model == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**params)
    elif model == 'RandomForestClassifier':
        return RandomForestClassifier(**params)
    elif model == 'Ridge':
        return Ridge(**params)
    elif model == 'SVR':
        return SVR(**params)
    elif model == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**params)
    elif model == 'RandomForestRegressor':
        return RandomForestRegressor(**params)


handlers = [logging.StreamHandler(sys.stdout)]
logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    level=logging.INFO,
                    handlers=handlers)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['ERROR_404_HELP'] = False
api = Api(app)
metrics = PrometheusMetrics(app)

MLFLOW_HOST = os.environ['MLFLOW_HOST']
MLFLOW_PORT = os.environ['MLFLOW_PORT']
# MLFLOW_HOST, MLFLOW_PORT = '0.0.0.0', 5050
MLFLOW_EXPERIMENT = 'hw3_exp'
mlflow.set_tracking_uri(f'http://{MLFLOW_HOST}:{MLFLOW_PORT}')
mlflow.set_experiment(MLFLOW_EXPERIMENT)
log.info(f'MLflow experiment {MLFLOW_EXPERIMENT} started')
client = MlflowClient()


class MLModelsDAO:
    def __init__(self):
        self.ml_models = {
            'ml_models': ['LogisticRegression', 'SVC', 'DecisionTreeClassifier',
                          'RandomForestClassifier', 'Ridge', 'SVR', 'DecisionTreeRegressor',
                          'RandomForestRegressor']}
        self.ml_models_all = {}
        self.counter = 0  # total number of fitted models

    def get(self, model_name, model_version):
        """
        Return predictions of the given model for the train and test set.
        Abort if the model was deleted or not fitted yet.

        :param str model_name: Model name to get predictions for
        :param str model_version: Model version to get predictions for
        :return: Predictions
        :rtype: dict
        """
        if model_name + '_' + model_version not in self.ml_models_all:
            log.error(f'Model {model_name} of version {model_version} does not exist\n'
                      f'Could not make predictions')
            return f'Model {model_name} of version {model_version} does not exist', 404
        if self.ml_models_all[model_name + '_' + model_version]['deleted']:
            log.error(f'Model {model_name} of version {model_version} was deleted\n'
                      f'Could not make predictions')
            return f'Model {model_name} of version {model_version} was deleted', 404

        log.info('Read data')
        with open(f'data/{model_name}_{model_version}_train.pkl', 'rb') as f:
            X_train = pickle.load(f).iloc[:, :-1]
        with open(f'data/{model_name}_{model_version}_test.pkl', 'rb') as f:
            X_test = pickle.load(f).iloc[:, :-1]

        log.info('Load model')
        estimator = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_version}')
        train_pred = list(np.round(estimator.predict(X_train), 2))
        test_pred = list(np.round(estimator.predict(X_test), 2))
        log.info(f'Predictions for model {model_name} of version {model_version} are made')

        return {'train_predictions': str(train_pred), 'test_predictions': str(test_pred)}

    def create(self, name, data, model, params, grid_search, param_grid):
        """
        Train the given model with given parameters on the given data (json) and
        calculate its performance metrics.
        Save train and test data into pkl files named by the model name and version.
        Log model and metrics for the mlflow run.
        If parameters (grid) are not set, use default values.
        Abort if the model is not available, data contain categorical features or estimator
        get an unexpected keyword argument.
        If no exceptions, append model name and parameters to models_dao.ml_models_all dictionary.

        :param str name: Model name in mlflow
        :param json data: Data to fit and test the model
        :param str model: Model to train
        :param str or dict params: Model parameters
        :param bool grid_search: Whether to perform grid search
        :param str or dict param_grid: Parameters grid for grid search
        :return: Fitting status
        :rtype: str
        """
        if model not in self.ml_models['ml_models']:
            log.error(f'Model {model} is not available\n Could not fit the model')
            return f'Can only train one of {self.ml_models} models', 404

        if params == 'default':
            params = {}
        if param_grid == 'default':
            param_grid = model2grid[model]
        if model == 'SVR' or model == 'SVC':
            params['probability'] = True

        with mlflow.start_run():
            log.info('Read data')
            df = pd.read_json(data)
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            log.info(f'Number of observations: {len(X)}')
            if any(X.dtypes == object):
                log.error('Data contains categorical features\nCould not fit the model')
                return 'Could not support categorical features', 400
            if y.dtype == object and model2task[model] == 'regression':
                log.error(f'{model} can only be used for regression tasks\n'
                          f'Could not fit the model')
                return f'{model} can only be used for regression tasks', 400
            elif y.dtype == float and model2task[model] == 'classification':
                log.error(f'{model} can only be used for classification tasks\n'
                          f'Could not fit the model')
                return f'{model} can only be used for classification tasks', 400

            log.info('Split data into train and test subsets')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                random_state=42)
            log.info(f'Size of training set: {len(X_train)} observations\n'
                     f'Size of testing set: {len(X_test)} observations')

            try:
                estimator = get_model(model, params)
            except TypeError as err:
                log.error(f'{model} got an unexpected keyword argument\nCould not fit the model')
                return f'{model} got an unexpected keyword argument {str(err).split()[-1]}', 400

            if grid_search:
                if not all([param in estimator.get_params().keys() for param in param_grid]):
                    log.error(f'Invalid parameters in grid for estimator {model}\n'
                              f'Could not fit the model')
                    return f'Invalid parameters for estimator {model}', 400
                log.info('Start grid search CV')
                gs = GridSearchCV(estimator, param_grid, cv=3, n_jobs=-1, verbose=2)
                start_time = time.time()
                gs.fit(X_train, y_train)
                gs_time = time.time() - start_time
                estimator = gs.best_estimator_
                gs_time = f'{np.round(gs_time / 60, 2)} min' if gs_time > 60 else \
                    f'{np.round(gs_time, 2)} sec'
                log.info(f'Finish grid search CV\nElapsed time: {gs_time}')

            log.info(f'Fit estimator with the following parameters: {estimator.get_params()}')
            mlflow.log_params(estimator.get_params())
            start_time = time.time()
            estimator.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_time = f'{np.round(training_time / 60, 2)} min' if training_time > 60 else \
                f'{np.round(training_time, 2)} sec'

            self.counter += 1
            log.info(f'{model} is fitted\nTraining time: {training_time}\n'
                     f'Total number of fitted models: {self.counter}')

            log.info('Calculate performance metrics')
            if model2task[model] == 'classification':
                y_train_pred, y_test_pred = estimator.predict(X_train), estimator.predict(X_test)
                y_train_pred_proba = estimator.predict_proba(X_train)
                y_test_pred_proba = estimator.predict_proba(X_test)
                average = 'binary' if len(set(y_train)) == 2 else 'macro'
                mlflow.log_metrics({
                    'train_accuracy': accuracy_score(y_train, y_train_pred),
                    'test_accuracy': accuracy_score(y_test, y_test_pred),
                    'train_precision': precision_score(y_train, y_train_pred, average=average),
                    'test_precision': precision_score(y_test, y_test_pred, average=average),
                    'train_recall': recall_score(y_train, y_train_pred, average=average),
                    'test_recall': recall_score(y_test, y_test_pred, average=average),
                    'train_f1': f1_score(y_train, y_train_pred, average=average),
                    'test_f1': f1_score(y_test, y_test_pred, average=average),
                    'train_auc': roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr'),
                    'test_auc': roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')})
            else:
                y_train_pred, y_test_pred = estimator.predict(X_train), estimator.predict(X_test)
                mlflow.log_metrics({
                    'train_rmse': mean_squared_error(y_train, y_train_pred, squared=False),
                    'test_rmse': mean_squared_error(y_test, y_test_pred, squared=False),
                    'train_mae': mean_absolute_error(y_train, y_train_pred),
                    'test_mae': mean_absolute_error(y_test, y_test_pred),
                    'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
                    'test_mape': mean_absolute_percentage_error(y_test, y_test_pred)})

            log.info('Save model')
            signature = infer_signature(X_train, estimator.predict(X_train))
            mlflow.sklearn.log_model(
                estimator, 'ml_model', signature=signature, registered_model_name=name)
            results = client.search_model_versions(f"name='{name}'")
            version = str(len(results))
            self.ml_models_all[name + '_' + version] = {'model': model,
                                                        'params': estimator.get_params(),
                                                        'retrained': False,
                                                        'deleted': False}

            log.info('Save train and test data')
            if self.counter == 1:
                list(map(os.unlink, (os.path.join('./data', f) for f in os.listdir('./data'))))
            with open(f'data/{name}_{version}_train.pkl', 'wb') as f:
                pickle.dump(df.loc[X_train.index], f)
            with open(f'data/{name}_{version}_test.pkl', 'wb') as f:
                pickle.dump(df.loc[X_test.index], f)

        return f'{model} is fitted and saved'

    def update(self, model_name, model_version, data):
        """
        Retrain the given model on a new training set and recalculate performance metrics.
        Save new train set into pkl files named by the model name and version.
        Log new model and metrics for the mlflow run.
        Abort if the model was deleted or not fitted yet.

        :param str model_name: Model name to retrain
        :param str model_version: Model version to retrain
        :param json data: Data to retrain
        :return: Re-fitting status
        :rtype: str
        """
        if model_name + '_' + model_version not in self.ml_models_all:
            log.error(f'Model {model_name} of version {model_version} does not exist\n'
                      f'Could not make predictions')
            return f'Model {model_name} of version {model_version} does not exist', 404
        if self.ml_models_all[model_name + '_' + model_version]['deleted']:
            log.error(f'Model {model_name} of version {model_version} was deleted\n'
                      f'Could not make predictions')
            return f'Model {model_name} of version {model_version} was deleted', 404

        log.info('Read data')
        with open(f'data/{model_name}_{model_version}_test.pkl', 'rb') as f:
            test = pickle.load(f)

        with mlflow.start_run():
            train_new = pd.read_json(data)
            X_train, y_train = train_new.iloc[:, :-1], train_new.iloc[:, -1]
            X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
            log.info(f'Size of training set: {len(X_train)} observations\n'
                     f'Size of testing set: {len(X_test)} observations')

            log.info('Load model')
            model = self.ml_models_all[model_name + "_" + model_version]["model"]
            estimator = mlflow.sklearn.load_model(f'models:/{model_name}/{model_version}')
            mlflow.log_params(estimator.get_params())
            log.info(f'Fit {model}')
            start_time = time.time()
            estimator.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_time = f'{np.round(training_time / 60, 2)} min' if training_time > 60 else \
                f'{np.round(training_time, 2)} sec'
            log.info(f'{model} is re-fitted\nTraining time: {training_time}\n')

            log.info('Calculate performance metrics')
            if model2task[model] == 'classification':
                y_train_pred, y_test_pred = estimator.predict(X_train), estimator.predict(X_test)
                y_train_pred_proba = estimator.predict_proba(X_train)
                y_test_pred_proba = estimator.predict_proba(X_test)
                average = 'binary' if len(set(y_train)) == 2 else 'macro'
                mlflow.log_metrics({
                    'train_accuracy': accuracy_score(y_train, y_train_pred),
                    'test_accuracy': accuracy_score(y_test, y_test_pred),
                    'train_precision': precision_score(y_train, y_train_pred, average=average),
                    'test_precision': precision_score(y_test, y_test_pred, average=average),
                    'train_recall': recall_score(y_train, y_train_pred, average=average),
                    'test_recall': recall_score(y_test, y_test_pred, average=average),
                    'train_f1': f1_score(y_train, y_train_pred, average=average),
                    'test_f1': f1_score(y_test, y_test_pred, average=average),
                    'train_auc': roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr'),
                    'test_auc': roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')})
            else:
                y_train_pred, y_test_pred = estimator.predict(X_train), estimator.predict(X_test)
                mlflow.log_metrics({
                    'train_rmse': mean_squared_error(y_train, y_train_pred, squared=False),
                    'test_rmse': mean_squared_error(y_test, y_test_pred, squared=False),
                    'train_mae': mean_absolute_error(y_train, y_train_pred),
                    'test_mae': mean_absolute_error(y_test, y_test_pred),
                    'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
                    'test_mape': mean_absolute_percentage_error(y_test, y_test_pred)})

            log.info('Save model')
            signature = infer_signature(X_train, estimator.predict(X_train))
            mlflow.sklearn.log_model(
                estimator, 'ml_model', signature=signature, registered_model_name=model_name)
            results = client.search_model_versions(f"name='{model_name}'")
            version = str(len(results))
            self.ml_models_all[model_name + '_' + version] = {'model': model,
                                                              'params': estimator.get_params(),
                                                              'retrained': model_version,
                                                              'deleted': False}

            log.info('Save new train data')
            with open(f'data/{model_name}_{model_version}_train.pkl', 'wb') as f:
                pickle.dump(train_new, f)

        return f'Model {model_name} of version {model_version} is re-fitted and saved'

    def delete(self, model_name, model_version):
        """
        Delete mlflow run of the given model.
        Abort if the model was deleted or not fitted yet.

        :param str model_name: Model name to delete pkl for
        :param str model_version: Model version to delete pkl for
        :return: None
        """
        if model_name + '_' + model_version not in self.ml_models_all:
            log.error(f'Model {model_name} of version {model_version} does not exist\n'
                      f'Could not make predictions')
            return f'Model {model_name} of version {model_version} does not exist', 404
        if self.ml_models_all[model_name + '_' + model_version]['deleted']:
            log.error(f'Model {model_name} of version {model_version} was deleted\n'
                      f'Could not make predictions')
            return f'Model {model_name} of version {model_version} was deleted', 404

        log.info('Find run id for the given model to delete')
        results = pd.DataFrame(
            [dict(i) for i in client.search_model_versions(f"name='{model_name}'")])
        run_id = results[results.version == model_version]['run_id'].values[0]
        mlflow.delete_run(run_id)
        self.ml_models_all[model_name + '_' + model_version]['deleted'] = True
        log.info(f'Model {model_name} of version {model_version} is deleted')


models_dao = MLModelsDAO()

common_counter = metrics.counter(
    'by_status_counter', 'Request count by status codes',
    labels={'status': lambda resp: resp.status_code}
)


@api.route('/ml_api')
class MLModels(Resource):

    @common_counter
    @metrics.counter('cnt_get_models', 'Number of ml models gets')
    def get(self):
        """
        Return a list of models available for training.

        :return: List of available models
        :rtype: list
        """
        return models_dao.ml_models

    @common_counter
    @metrics.summary('cnt_trains', 'Number of fitting tasks per model name',
                     labels={'model_name': lambda: request.json['name'],
                             'status': lambda resp: resp.status_code})
    @metrics.summary('cnt_estimator_uses', 'Number of fitting tasks per estimator',
                     labels={'estimator': lambda: request.json['model'],
                             'status': lambda resp: resp.status_code})
    def post(self):
        """
        Train the given model with given parameters on the given data (json) and
        calculate its performance metrics.
        Save train and test data into pkl files named by the model name and version.
        Log model and metrics for the mlflow run.
        If parameters (grid) are not set, use default values.
        Abort if the model is not available, data contain categorical features or estimator
        get an unexpected keyword argument.
        If no exceptions, append model name and parameters to models_dao.ml_models_all dictionary.

        :return: Fitting task status
        :rtype: str
        """
        json_ = request.json
        name = json_['name']
        data = json_['data']
        model = json_['model']
        params = json_.get('params', 'default')
        grid_search = json_.get('grid_search', False)
        param_grid = json_.get('param_grid', 'default')
        log.info(f'Fit {model}')
        return models_dao.create(name, data, model, params, grid_search, param_grid)


@api.route('/ml_api/<model_name>/<model_version>')
class MLModelsID(Resource):

    @common_counter
    @metrics.counter('cnt_get_preds', 'Number of prediction tasks per model name',
                     labels={'model_name': lambda: request.view_args['model_name'],
                             'status': lambda resp: resp.status_code})
    def get(self, model_name, model_version):
        """
        Return predictions of the given model for the train and test set.
        Abort if the model was deleted or not fitted yet.

        :param str model_name: Model name to get predictions for
        :param str model_version: Model version to get predictions for
        :return: Predictions
        :rtype: dict
        """
        log.info(f'Get predictions for model {model_name} of version {model_version}')
        return models_dao.get(model_name, model_version)

    @common_counter
    @metrics.summary('cnt_retrains', 'Number of retrains per model name',
                     labels={'model_name': lambda: request.view_args['model_name'],
                             'status': lambda resp: resp.status_code})
    def put(self, model_name, model_version):
        """
        Retrain the given model on a new training set and recalculate performance metrics.
        Save new train set into pkl files named by the model name and version.
        Log new model and metrics for the mlflow run.
        Abort if the model was deleted or not fitted yet.

        :param str model_name: Model name to retrain
        :param str model_version: Model version to retrain
        :return: Re-fitting task status
        :rtype: str
        """
        data = request.json
        log.info(f'Retrain model {model_name} of version {model_version} on a new data')
        return models_dao.update(model_name, model_version, data)

    @common_counter
    @metrics.counter('cnt_deletes', 'Number of deletions per model name',
                     labels={'model_name': lambda: request.view_args['model_name'],
                             'status': lambda resp: resp.status_code})
    def delete(self, model_name, model_version):
        """
        Delete mlflow run of the given model.
        Abort if the model was deleted or not fitted yet.

        :param str model_name: Model name to delete pkl file for
        :param str model_version: Model version to delete pkl file for
        :return: Nothing
        :rtype: str
        """
        log.info(f'Delete model {model_name} of version {model_version}')
        models_dao.delete(model_name, model_version)
        return '', 204


@api.route('/ml_api/all_models')
class MLModelsAll(Resource):

    @common_counter
    @metrics.counter('cnt_get_all_models', 'Number of all models gets')
    def get(self):
        """
        Return a dictionary of all fitted models and their parameters.

        :return: Dictionary of all fitted models
        :rtype: dict
        """
        log.info(f'Total number of fitted models: {len(models_dao.ml_models)}')
        return models_dao.ml_models_all


if __name__ == '__main__':
    log.info('App started')
    app.run(host=os.environ['HOST'],
            port=os.environ['PORT'])
    # app.run(host='0.0.0.0', port=8080, debug=False)
    log.info('App finished')

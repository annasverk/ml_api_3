import unittest
import pickle
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp
from flask_app import app as test_app

boston_data = load_boston()
boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['PRICE'] = boston_data.target
boston_json = boston_df.to_json()

iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['class'] = iris_data.target
iris_json = iris_df.to_json()

boston_df_cat = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df_cat['category'] = 'a'
boston_df_cat['PRICE'] = boston_data.target
boston_json_cat = boston_df_cat.to_json()

client = MlflowClient()


class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.client = test_app.app.test_client()

    def test_get_ml_models(self):
        """
        Test GET request. Must return a dictionary with ml_models key.
        """
        response = self.client.get('/ml_api')
        self.assertListEqual(['ml_models'], list(response.json.keys()))
        self.assertEqual(response.status_code, 200)

    def test_get_wrong_model(self):
        """
        Test POST request. Must return 404 error if model is not one of
        LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier,
        Ridge, SVR, DecisionTreeRegressor, RandomForestRegressor.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegression', 'data': boston_json, 'name': 'rf'})
        self.assertIn(b'Can only train one of', response.data)
        self.assertEqual(response.status_code, 404)

    def test_categorical_features(self):
        """
        Test POST request. Must return 400 error if data contain categorical features.
        """
        response = self.client.post(
            '/ml_api',
            json={'model': 'RandomForestRegressor', 'data': boston_json_cat, 'name': 'rf'})
        self.assertIn(b'Could not support categorical features', response.data)
        self.assertEqual(response.status_code, 400)

    def test_wrong_model_task(self):
        """
        Test POST request. Must return 400 error if model does not correspond to task
        defined by the target type.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestClassifier', 'data': boston_json, 'name': 'rf'})
        self.assertIn(b'RandomForestClassifier can only be used for classification tasks',
                      response.data)
        self.assertEqual(response.status_code, 400)

    def test_wrong_parameter(self):
        """
        Test POST request. Must return 400 error if keyword argument is wrong for this estimator.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'DecisionTreeRegressor', 'data': boston_json, 'name': 'rf',
                             'params': {'n_estimators': 200}})
        self.assertIn(b'DecisionTreeRegressor got an unexpected keyword argument', response.data)
        self.assertEqual(response.status_code, 400)

    def test_wrong_grid_parameter(self):
        """
        Test POST request. Must return 400 error if grid parameter is wrong for this estimator.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'DecisionTreeRegressor', 'data': boston_json, 'name': 'rf',
                             'grid_search': True, 'param_grid': {'n_estimators': [100, 200]}})
        self.assertIn(b'Invalid parameters for estimator DecisionTreeRegressor', response.data)
        self.assertEqual(response.status_code, 400)

    def test_post(self):
        """
        Test POST request. Must return 200 status and message about successful fitting.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        self.assertIn(b'RandomForestRegressor is fitted and saved', response.data)
        self.assertEqual(response.status_code, 200)

    def test_get_all_models(self):
        """
        Test GET request. Must return a dictionary of len at least 1 after POST request sent.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        response = self.client.get('/ml_api/all_models')
        self.assertGreater(len(response.json), 0)
        self.assertEqual(response.status_code, 200)

    def test_get_predictions(self):
        """
        Test GET request. Must return train and test predictions for the given model.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        model_name, model_version = last_model.split('_')[0], last_model.split('_')[1]
        response = self.client.get(f'/ml_api/{model_name}/{model_version}')
        self.assertIsNotNone(response.get_data(as_text=True))
        self.assertEqual(response.status_code, 200)

    def test_wrong_model_name(self):
        """
        Test GET/POST/PUT request. Must return 404 error if the given model
        of the given version does not exist yet.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        response = self.client.get('/ml_api/rf_model/1')
        self.assertIn(b'Model rf_model of version 1 does not exist', response.data)
        self.assertEqual(response.status_code, 404)

    def test_deleted_model(self):
        """
        Test GET/POST/PUT request. Must return 404 error if the given model
        of the given version was already deleted.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        model_name, model_version = last_model.split('_')[0], last_model.split('_')[1]
        response = self.client.delete(f'/ml_api/{model_name}/{model_version}')
        response = self.client.get(f'/ml_api/{model_name}/{model_version}')
        self.assertIn(b'was deleted', response.data)
        self.assertEqual(response.status_code, 404)

    def test_delete(self):
        """
        Test DELETE request. Must return 204 status if the given model is successfully deleted.
        Also check that 'deleted' flag in 'all_models' dictionary changed to True.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        model_name, model_version = last_model.split('_')[0], last_model.split('_')[1]
        response = self.client.delete(f'/ml_api/{model_name}/{model_version}')
        self.assertEqual(response.status_code, 204)
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        self.assertEqual(all_models[last_model]['deleted'], True)

    def test_put(self):
        """
        Test PUT request. Must return 200 status and message about successful re-fitting.
        Also check that 'retrained' value in 'all_models' dictionary changed to version
        of the model that has been retrained.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        model_name, model_version = last_model.split('_')[0], last_model.split('_')[1]
        response = self.client.put(f'/ml_api/{model_name}/{model_version}',
                                   json=boston_df.sample(100).to_json())
        self.assertIn(b'is re-fitted and saved', response.data)
        self.assertEqual(response.status_code, 200)
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        self.assertEqual(all_models[last_model]['retrained'], model_version)

    def test_default_params(self):
        """
        Test POST request. Must return a dictionary of default sklearn estimator parameters
        if 'params' is not set in a send json data.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        results = [dict(i) for i in client.search_model_versions(f"name='rf'")]
        last_run = results[-1]['run_id']
        params = client.get_run(last_run).data.params
        self.assertDictEqual(
            {'bootstrap': 'True', 'ccp_alpha': '0.0', 'criterion': 'mse', 'max_depth': 'None',
             'max_features': 'auto', 'max_leaf_nodes': 'None', 'max_samples': 'None',
             'min_impurity_decrease': '0.0', 'min_impurity_split': 'None', 'min_samples_leaf': '1',
             'min_samples_split': '2', 'min_weight_fraction_leaf': '0.0', 'n_estimators': '100',
             'n_jobs': 'None', 'oob_score': 'False', 'random_state': 'None', 'verbose': '0',
             'warm_start': 'False'}, params)

    def test_train_size(self):
        """
        Test POST request. Check the size of the training set knowing the proportion
        of the dataset to include in the train split.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        with open(f'data/{last_model}_train.pkl', 'rb') as f:
            train = pickle.load(f)
        self.assertEqual(train.shape, (354, 14))

    def test_train_test_distributions(self):
        """
        Test POST request. Perform the two-sample Kolmogorov-Smirnov test for goodness of fit.
        Check the null hypothesis that the two distributions (train and test targets) are identical.
        """
        response = self.client.post(
            '/ml_api', json={'model': 'RandomForestRegressor', 'data': boston_json, 'name': 'rf'})
        all_models = self.client.get('/ml_api/all_models').json
        last_model = list(all_models.keys())[-1]
        with open(f'data/{last_model}_train.pkl', 'rb') as f:
            y_train = pickle.load(f).iloc[:, -1]
        with open(f'data/{last_model}_test.pkl', 'rb') as f:
            y_test = pickle.load(f).iloc[:, -1]
        self.assertGreater(ks_2samp(y_train, y_test).pvalue, 0.01)


if __name__ == '__main__':
    unittest.main()

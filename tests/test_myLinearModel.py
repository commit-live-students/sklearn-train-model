from unittest import TestCase


class TestMyLinearModel(TestCase):
    def test_myLinearModel(self):
        import sklearn
        from sklearn import datasets
        from build import myLinearModel

        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target

        mse, model = myLinearModel(X, y)

        self.assertAlmostEqual(mse, 2859.6903987680657)
        self.assertIsInstance(model, sklearn.linear_model.base.LinearRegression)
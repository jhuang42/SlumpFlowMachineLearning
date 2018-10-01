from sklearn import linear_model
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt


# Class to contain all the regression functions we will be using (OLS, Ridge, Lasso, R^2) 
class RegressionSuite:
    # An unregularized regression of slump flow values against the seven explanatory variables.
    # What is the R-squared? Plot a graph evaluating each regression.
    def ols_regression(self, x_df, y_df, x_test, y_test):
        ols_reg = linear_model.LinearRegression()
        ols_reg.fit(x_df[x_df.columns.values], y_df['flow'])
        y_pred = ols_reg.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rss = self.rss_function(y_pred, y_test)
        result = [mse]
        result.extend([rss])
        result.extend([ols_reg.intercept_])
        result.extend([ols_reg.coef_])
        print('[mse, rss, intercept, coefficients' + str(x_df.columns.values) + ']')
        print(result)
        print("R2 score:")
        print(metrics.r2_score(y_test, y_pred))
        return result

    def ridge_regression(self, x_df, y_df, x_test, y_test, alpha):
        ridge_reg = linear_model.Ridge(alpha=alpha, normalize=True)
        ridge_reg.fit(x_df[x_df.columns.values], y_df['flow'])
        y_pred = ridge_reg.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rss = self.rss_function(y_pred, y_test)
        result = [mse]
        result.extend([rss])
        result.extend([ridge_reg.intercept_])
        result.extend([ridge_reg.coef_])
        print('[mse, rss, intercept, coefficients' + str(x_df.columns.values) + ']')
        print(result)
        print("R2 score:")
        print(metrics.r2_score(y_test, y_pred))

        # Task 2
        print("Computing regularization path using the LARS ...")
        x_test[x_test.columns] = x_test[x_test.columns].astype(float)
        alphas, _, coefs = linear_model.lars_path(x_test.values, y_pred, method='ridge', verbose=True)
        xx = numpy.sum(numpy.abs(coefs.T), axis=1)
        xx /= xx[-1]
        plt.plot(xx, coefs[0].T, marker='o', label="cement")
        plt.plot(xx, coefs[1].T, marker='o', label="coarse")
        plt.plot(xx, coefs[2].T, marker='o', label="fine")
        plt.plot(xx, coefs[3].T, marker='o', label="fly_ash")
        plt.plot(xx, coefs[4].T, marker='o', label="slag")
        plt.plot(xx, coefs[5].T, marker='o', label="sp")
        plt.plot(xx, coefs[6].T, marker='o', label="water")
        plt.title('Ridge Path')
        plt.xlabel('|coef| / max|coef|')
        plt.ylabel('Coefficients')
        plt.axis('tight')
        # Place a legend to the right of this smaller subplot.
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        plt.show()
        plt.close()

        return result

    def lasso_regression(self, x_df, y_df, x_test, y_test, alpha):
        lasso_reg = linear_model.Lasso(alpha=alpha, normalize=True)
        lasso_reg.fit(x_df[x_df.columns.values], y_df['flow'])
        y_pred = lasso_reg.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rss = self.rss_function(y_pred, y_test)
        result = [mse]
        result.extend([rss])
        result.extend([lasso_reg.intercept_])
        result.extend([lasso_reg.coef_])
        print('[mse, rss, intercept, coefficients' + str(x_df.columns.values) + ']')
        print(result)
        print("R2 score:")
        print(metrics.r2_score(y_test, y_pred))

        # Task 2
        print("Computing regularization path using the LARS ...")
        x_test[x_test.columns] = x_test[x_test.columns].astype(float)
        alphas, _, coefs = linear_model.lars_path(x_test.values, y_pred, method='lasso', verbose=True)
        xx = numpy.sum(numpy.abs(coefs.T), axis=1)
        xx /= xx[-1]
        plt.plot(xx, coefs[0].T, marker='o', label="cement")
        plt.plot(xx, coefs[1].T, marker='o', label="coarse")
        plt.plot(xx, coefs[2].T, marker='o', label="fine")
        plt.plot(xx, coefs[3].T, marker='o', label="fly_ash")
        plt.plot(xx, coefs[4].T, marker='o', label="slag")
        plt.plot(xx, coefs[5].T, marker='o', label="sp")
        plt.plot(xx, coefs[6].T, marker='o', label="water")
        plt.title('LASSO Path')
        plt.xlabel('|coef| / max|coef|')
        plt.ylabel('Coefficients')
        plt.axis('tight')
        # Place a legend to the right of this smaller subplot.
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        plt.show()
        plt.close()

        return result

    @staticmethod
    def get_ridge_score(x_df, y_df, x_valid, y_valid):
        ridge_reg = linear_model.Lasso(alpha=1, normalize=True)
        ridge_reg.fit(x_df[x_df.columns.values], y_df['flow'])
        return ridge_reg.score(x_valid, y_valid)

    @staticmethod
    def get_lasso_score(x_df, y_df, x_valid, y_valid):
        lasso_reg = linear_model.Ridge(alpha=1, normalize=True)
        lasso_reg.fit(x_df[x_df.columns.values], y_df['flow'])
        return lasso_reg.score(x_valid, y_valid)

    @staticmethod
    def rss_function(y_pred, y_df):
        return sum((y_pred - numpy.array(y_df['flow'], numpy.float64))**2)

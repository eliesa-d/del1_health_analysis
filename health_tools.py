import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class HealthAnalyzer:
    """ Klass där jag samlar funktioner för att analysera datan. Innehåller enkla metoder för statistik, grafer och regression"""

    def __init__(self, data):
        """ Tar emot en dataframe och sparar den i objektet"""
        self.data = data

    def summary_stats(self):
        """ Snabb översikt av datan: n, medel, median, min och max """
        return self.data.describe().T[['count', 'mean', '50%', 'min', 'max']].rename(
            columns={'count': 'n', '50%': 'median'}
        )

    def plot_bp_distribution(self):
        """ Ritar ett histogram över systoliskt blodtryck"""
        plt.hist(self.data["systolic_bp"], bins=20, color="lightblue", edgecolor="black")
        plt.title("Fördelning av systoliskt blodtryck")
        plt.xlabel("Systoliskt blodtryck (mmHg)")
        plt.ylabel("Antal personer")
        plt.show()

    def correlation(self, col1, col2):
        """ Räknar ut korrelation mellan två valfria kolumner"""
        return np.corrcoef(self.data[col1], self.data[col2])[0, 1]

    def simple_regression(self, x_col, y_col):
        """ Enkel linjär regression för att se sambandet mellan två variabler. Returnerar lutning, intercept och R2"""
        x = self.data[x_col]
        y = self.data[y_col]
        b, a = np.polyfit(x, y, 1)
        y_pred = a + b * x
        r2 = np.corrcoef(x, y)[0, 1] ** 2

        plt.scatter(x, y, color="lightblue", label="Data")
        plt.plot(x, y_pred, color="red", label=f"y = {a:.2f} + {b:.2f}x")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Linjär regression: {y_col} vs {x_col}")
        plt.legend()
        plt.show()

        return {"intercept": a, "slope": b, "r2": r2}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class HealthAnalyzer:
    """En enkel klass för att samla analysfunktioner för hälsostudien.
    Den används för att hålla koden mer organiserad i Del 2.
    """

    def __init__(self, data: pd.DataFrame):
        """Tar in en DataFrame och sparar den i objektet.

        Parametrar
        ----------
        data : pd.DataFrame
            Hälsodatat från studien.
        """
        self.df = data

    def basic_stats(self, column):
        """Returnerar medelvärde, median, min och max för en vald kolumn.

        Parametrar
        ----------
        column : str
            Namnet på kolumnen som ska analyseras.

        Returnerar
        ----------
        dict
            En ordbok med medelvärde, median, min och max.
        """
        col = self.df[column]
        return {
            "mean": col.mean(),
            "median": col.median(),
            "min": col.min(),
            "max": col.max()
        }

    def plot_relation(self, x, y):
        """Ritar ett enkelt scatterplot mellan två variabler.

        Parametrar
        ----------
        x : str
            Variabel på x-axeln.
        y : str
            Variabel på y-axeln.
        """
        plt.scatter(self.df[x], self.df[y], alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{y} som funktion av {x}")
        plt.show()

    def simple_regression(self, x_var, y_var):
        """Kör en enkel linjär regression med scikit-learn.
        Returnerar lutning, intercept och R².

        Parametrar
        ----------
        x_var : str
            Oberoende variabel.
        y_var : str
            Beroende variabel.

        Returnerar
        ----------
        dict
            Lutning, intercept och R²-värde.
        """
        X = self.df[[x_var]].values
        y = self.df[y_var].values

        model = LinearRegression()
        model.fit(X, y)

        return {
            "slope": model.coef_[0],
            "intercept": model.intercept_,
            "r2": model.score(X, y)
        }
    def multi_regression_age_weight_bp(self):
        """Multipel regression där ålder och vikt används för att förklara
        systoliskt blodtryck.

        Returnerar
        ----------
        dict
            Koefficienter, intercept, R² och predikterade värden.
        """
        X = self.df[["age", "weight"]].values
        y = self.df["systolic_bp"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        return {
            "coef_age": model.coef_[0],
            "coef_weight": model.coef_[1],
            "intercept": model.intercept_,
            "r2": model.score(X, y),
            "y_pred": y_pred,
        }
    def plot_disease_by_sex(self):
        """Ritar ett stapeldiagram över andelen sjukdom per kön.

        Returnerar
        ----------
        None
            Visar bara figuren.
        """
        # antar att disease är kodad som 0/1
        prevalence = self.df.groupby("sex")["disease"].mean()

        plt.bar(prevalence.index, prevalence.values)
        plt.xlabel("Kön")
        plt.ylabel("Andel med sjukdom")
        plt.title("Andel sjukdom per kön")
        plt.ylim(0, 1)  # andel mellan 0 och 1
        plt.show()

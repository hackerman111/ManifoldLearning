"""Population gradient geometry of the study links under standard Gaussian X."""

import csv

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from experiments.mave_grassmann_grid import OUT


def expectation(function):
    return quad(lambda z: function(z)*np.exp(-z*z/2)/np.sqrt(2*np.pi),
                -12, 12, epsabs=1e-12)[0]


def main():
    v_sin = (1-np.exp(-2))/2
    e_cos = np.exp(-.5)
    e_cos2 = (1+np.exp(-2))/2
    sech2 = lambda z: 4*expit(2*z)*expit(-2*z)
    e_sech2 = expectation(sech2)
    e_sech4 = expectation(lambda z: sech2(z)**2)
    var_tanh = 1-e_sech2
    rows = []
    for link in ("additive", "sine", "quadratic"):
        for m in (1, 2, 3, 5):
            if link == "additive":
                means = np.array([e_cos]+([0.] if m >= 2 else [])+[.35*e_sech2]*max(0,m-2))
                seconds = np.array([e_cos2]+([1.] if m >= 2 else [])+[.35**2*e_sech4]*max(0,m-2))
                variance = v_sin + (.5 if m >= 2 else 0) + .35**2*var_tanh*max(0,m-2)
                opg = (np.diag(seconds-means**2)+np.outer(means, means))/variance
            elif link == "sine":
                variance = m*v_sin
                opg = ((e_cos2-e_cos**2)*np.eye(m)+e_cos**2*np.ones((m,m)))/variance
            else:
                variance = 2*m
                opg = (4/variance)*np.eye(m)
            eig = np.linalg.eigvalsh(opg)
            assert eig[0] > 0
            rows.append(dict(link=link, m=m, raw_signal_variance=variance,
                             opg_lambda_min=eig[0], opg_lambda_max=eig[-1],
                             opg_condition=eig[-1]/eig[0]))
    with (OUT / "signal_geometry.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Population-standardized E[gradient f gradient f.T], not empirical ADP statistics:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

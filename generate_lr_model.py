import pandas as pd
import numpy as np
import sklearn.linear_model
import os
import pickle

def generate_models(dataset):
    """
    Generates a linear regression model for:

        Let V0, V1, V2 be the raw sensors signals
        Let B0, B1, B2 be the background signals

        Let Ui = Vi - Bi

        The model fits:

            ln(CH4 ppm) = A*U0 + B*U1 + C*U2 + D

    """
    # Filter data points with zero CH4 ppm (since log 0 is undefined)
    dataset = dataset[dataset['ch4 / ppm'] > 0 ]

    # Subtract background
    signals = np.array([dataset['v0 / volts']-dataset['bg v0 / volts'],
                        dataset['v1 / volts']-dataset['bg v1 / volts'],
                        dataset['v2 / volts']-dataset['bg v2 / volts']])

    concentration = dataset['ch4 / ppm']

    # Fit linear model
    lr_model = sklearn.linear_model.LinearRegression()
    lr_model.fit(np.transpose(signals), np.log(concentration))

    # Calculate R^2
    lr_r2 = lr_model.score(np.transpose(signals), np.log(concentration))
    print("Linear Regression Model: R^2 = %f"%lr_r2)

    return lr_model

if __name__ == '__main__':

    methane_lo_dataset = pd.read_csv(os.path.join('calibration_data', '20221003_nglo.csv'))
    lr_model = generate_models(methane_lo_dataset)

    with open(os.path.join('models', 'lr_model_nglo.bin'), 'wb') as mdl_out:
        pickle.dump(lr_model, mdl_out)
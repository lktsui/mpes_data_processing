import pickle
import numpy as np
import os
import sklearn

def predict_concentration(signal_raw, background, model):
    """
    :param signal_raw: a numpy array or a list of the
                        form [v0, v1, v2] representing the raw signal obtained from the sensor
    :param background: a numpy array or a list of the for
                        form [v0, v1, v2] representing the background signal collected in ambient air / zero air
    :param model: linear regression model (sklearn)
    :return: lnch4: natural log of ch4 concentration detected
    """

    # Convert to numpy array and reshape to format regression model likes
    signal_raw = np.array(signal_raw)
    signal_raw = np.transpose(signal_raw.reshape(-1, 1))
    # Convert to numpy array and reshape to format regression model likes
    background = np.array(background)
    background = np.transpose(background.reshape(-1,1))

    # Calculate Signal - Background
    signal_delta = signal_raw-background

    lnch4 = model.predict(signal_delta)[0]
    return lnch4

def main():
    # All signal values should be given in VOLTS
    signal_raw = [-0.07279103448275863, 0.016126896551724137, -0.00020451034482758622]

    # Capture a background signal prior to inference
    background = [ -0.02813655172413793, 0.018223103448275865, 0.018223103448275865]

    # Natural Gas - Low Ethane is the closest mixture to what is being tested at METEC. Load this model.
    with open(os.path.join('models', 'lr_model_nglo.bin'), 'rb') as mdl_file:
        lr_model = pickle.load(mdl_file)

    # Linear Regression
    lnch4 = predict_concentration(signal_raw, background, lr_model)
    predicted_ch4_ppm = np.exp(lnch4)
    true_ppm = 155
    error = 100*(predicted_ch4_ppm-true_ppm)/true_ppm

    print("Linear Regression Model")
    print("True PPM: %f"%true_ppm)
    print("Predicted PPM: %f"%predicted_ch4_ppm)
    print("Error: %f percent"%error)

if __name__ == '__main__':
    main()
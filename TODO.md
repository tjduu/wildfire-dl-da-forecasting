### Project TODO list:

* `[objective 1]`: build a sequential forecasting model (LSTM, RNN, ConvLSTM) that takes in image at time t and forecasts t+1(name) [**X**]
* Data Loader / Dataset
* Split of fires (labeled)
* `[objective 2]`: build a generative model (VAE, VQ-VAE) that generates an image (no necessarily in the next time step)
* `[objective 2]`: create dataset for generative model
* `[objective 2]`: create loader for generative model
* `[objective 3]`: write data assimilation class `class DA(model, sensor, R, B)` OR `class DA(model, model_data, sensor_data, R, B)`

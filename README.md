# opampDL_phaseMargin
on this project I took a two stage opamp. Sweeping the M of the diff pair, active load, Mos output stage, C miller and R I tried to predict the phase margin
( very few data should add more!!!)


(some parameters to sweep the l, more steps on C miller, R, and of course try with other Ibias

9/9 - 0s - loss: 0.1570 - mean_squared_error: 0.1570 - val_loss: 0.1329 - val_mean_squared_error: 0.1329 - 38ms/epoch - 4ms/step
Epoch 398/400

9/9 - 0s - loss: 0.1766 - mean_squared_error: 0.1766 - val_loss: 0.1176 - val_mean_squared_error: 0.1176 - 56ms/epoch - 6ms/step
Epoch 399/400

9/9 - 0s - loss: 0.1836 - mean_squared_error: 0.1836 - val_loss: 0.0798 - val_mean_squared_error: 0.0798 - 35ms/epoch - 4ms/step
Epoch 400/400

9/9 - 0s - loss: 0.1475 - mean_squared_error: 0.1475 - val_loss: 0.0939 - val_mean_squared_error: 0.0939 - 33ms/epoch - 4ms/step


Prediction:  63.999733  true value:  phaseMargin    64.162903

Prediction:  65.15351  true value:  phaseMargin    64.819298

Prediction:  67.10218  true value:  phaseMargin    67.406097

Prediction:  70.18443  true value:  phaseMargin    69.774002

Prediction:  66.137085  true value:  phaseMargin    66.153198

Prediction:  66.04474  true value:  phaseMargin    66.522301

Prediction:  75.371315  true value:  phaseMargin    75.875198

Prediction:  66.6088  true value:  phaseMargin    66.206596

Prediction:  77.04142  true value:  phaseMargin    77.155296

Prediction:  65.44346  true value:  phaseMargin    65.222198


![Figure_1](https://github.com/georgekasa/opampDL_phaseMargin/assets/79354220/1905458f-82db-4ef5-ba94-3ebbbf453abc)




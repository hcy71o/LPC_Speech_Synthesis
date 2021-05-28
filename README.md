# LPC_Speech_Synthesis
### - Predict current sample value from past p samples 

### - Useful for compression or transmission by encoding only residual and linear prediction coefficients

# Overview
<img src = "./Figures/Overview.png" width="90%">

# Speech Analysis
### - Get unvoiced/voiced flag information, LPC coefficients, and detect pitch
    Speech_analysis.ipynb

## *Vocal tract modelling (AR modelling)
<img src = "./Figures/Linear_prediction.jpeg" width="90%">

### Autocorrelation Method

<img src = "./Figures/ACR.png" width="50%">

# Speech Synthesis
#### - Synthesize speech using LPC information and excitation
    Speech_synthesis.ipynb


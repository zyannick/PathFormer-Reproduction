# PathFormer-Reproduction
Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting

## Modules

### Multi-scale router

The multi-scale router play the role of adaptive multi-scale modeling for time series forecasting:

- Select specific patch sizes for dividing the times series data based on the input data characteristics
- Controls the extraction of multiscale features by determining which patch size
- Enhances the model's capabilty to grasp temporal dynamics by incorporating trend and seasonality

Here is high level view of the multi scale router module

<img src="draws/PathFormer-MultiscaleRouter.drawio.svg" alt="MultiscaleRouter" width="300"/>


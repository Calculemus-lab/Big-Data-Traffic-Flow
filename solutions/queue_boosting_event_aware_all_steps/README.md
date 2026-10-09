# Event-aware queue forecasting without an onset horizon restriction

This is an ablation of `queue_boosting_event_aware`. It keeps its features,
selector-matched onset training, LightGBM models, and ongoing forecasts. For
onset windows, it allows thresholded predictions at all six forecast steps;
the onset-specific classifier still replaces the T+30 probability. Historical
threshold calibration uses the same unrestricted decoder.

Compare its Task 2-only validation score with the T+30-only candidate. All
other task outputs and model settings should remain identical.

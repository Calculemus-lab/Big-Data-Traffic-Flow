# Event-aware queue forecasting without an onset horizon restriction

This is an ablation of `queue_boosting_event_aware`. It keeps its features,
selector-matched onset training, LightGBM models, and ongoing forecasts. For
onset windows, it allows thresholded predictions at all six forecast steps;
the onset-specific classifier still replaces the T+30 probability. Historical
threshold calibration uses the same unrestricted decoder.

Its official Task 2-only validation score is approximately 0.70397, compared
with 0.71773 for the T+30-only candidate. The T+30 restriction performs better
on the public validation split, while this variant scores higher on the small
random local proxy case.

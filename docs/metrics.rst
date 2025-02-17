.. _metrics:

#######
Metrics
#######

To evaluate the performance of different trackers, metrics were defined.
These metrics either evaluate the tracking of the detected people or they evaluate the detections themselves.

Detection Metrics
*****************

- IoU: intersection over union
- OKS: object keypoint similarity
- DetA: detection accuracy

Tracking Metrics
****************

.. _metrics_mota:

Multi-Object Tracking Accuracy
==============================

The multi-object tracking accuracy or MOTA was the gold standard for tracking.
But it puts more weight on detection accuracy than on tracking accuracy, thus resulting in imprecise tracking evaluation metrics.

TODO formula
TODO how is MOTA computed for each dataset

.. _metrics_hota:

Higher-Order Tracking Accuracy
==============================

THe higher order tracking accuracy or HOTA was developed to balance between the detection and tracking accuracy.
It can only be computed over the whole video sequence after all predictions were made.
A more in depth explanation of the HOTA can be found _`<here> https://autonomousvision.github.io/hota-metrics/` .

TODO formula
TODO how is HOTA computed for each dataset


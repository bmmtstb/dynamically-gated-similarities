.. _metrics:

#######
Metrics
#######

To evaluate the performance of different trackers, metrics were defined.
These metrics either evaluate the tracking of the detected people or they evaluate the detections themselves.

Detection Metrics
*****************

- IoU: intersection over union between two bounding boxes

  .. math::

  	\text{IoU} &= \frac{\text{Area of Overlap}}{\text{Area of Union}}
- OKS: object keypoint similarity between two sets of keypoints (see e.g. `here <https://learnopencv.com/object-keypoint-similarity/>`_)

  .. math::

	\texttt{OKS} &= \frac{\sum_i \texttt{KS}_i \cdot \delta \left( v_i > 0 \right)}{\sum_i \delta \left( v_i > 0 \right)} = \frac{\texttt{key point similarity}}{\texttt{nof labeled}}\\
	\texttt{KS}_i &= \exp{\left( \frac{- d^2_i}{2s^2 k^2_i} \right)}
- LocA: location accuracy

  .. math::

	\texttt{LocA} &= \frac{1}{|\texttt{TP}|}\sum_{c\in \texttt{TP}}\texttt{IoU}\left( c \right)
- DetA: detection accuracy

  .. math::

	\texttt{DetA} &= \frac{|\texttt{TP}|}{|\texttt{TP}| + |\texttt{FP}| + |\texttt{FN}|}

Tracking Metrics
****************

The performance of a tracker is evaluated after the whole video sequence was processed and can be evaluated using the following metrics:

.. _metrics_mota:

Multi-Object Tracking Accuracy
==============================

The multi-object tracking accuracy or :math:`\texttt{MOTA}` was the gold standard for tracking.
But it puts more weight on detection accuracy than on tracking accuracy, thus resulting in imprecise tracking evaluation metrics.

.. math::

	\texttt{MOTA} &= 1 - \frac{|\texttt{FN}| + |\texttt{FP}| + |\texttt{IDSW}|}{|\texttt{TP}|}
	\quad
	\begin{cases}
		\texttt{FN}\text{: nof false negatives}\\
		\texttt{FP}\text{: nof false positives}\\
		\texttt{IDSW}\text{: nof ID switches}\\
		\texttt{TP}\text{: nof true positives}
	\end{cases}

Within this code, the :math:`\texttt{MOTA}` is computed using the evaluation tools of the respective datasets.
For the DanceTrack dataset the DanceTrack internal `TrackEval <https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval>`_ package was used.
For the PoseTrack21 dataset, the `poseval <https://github.com/leonid-pishchulin/poseval>`_ package was used to be able to evaluate more joint-dependent parameters of the PoseTrack21 dataset.
All modules are (or at least can be) loaded as submodules while setting up the DGS framework.

.. _metrics_hota:

Higher-Order Tracking Accuracy
==============================

THe higher order tracking accuracy or :math:`\texttt{HOTA}` was developed to balance between the detection and tracking accuracy.
It can only be computed over the whole video sequence after all predictions were made.
A more in depth explanation of the HOTA can be found on `autonomousvision.github.io <https://autonomousvision.github.io/hota-metrics/>`_ .

.. math::

	\texttt{HOTA}_\alpha &= \sqrt{\texttt{DetA}_\alpha \cdot \texttt{AssA}_\alpha}\\
	&= \sqrt{\frac{\sum_{c \in \lbrace \texttt{TP} \rbrace} \mathcal{A} \left( c \right)}{|\texttt{TP}| + |\texttt{FN}| + |\texttt{FP}|}}\\
	\\
	\texttt{Ass-IoU} &= \frac{|\texttt{TPA}|}{|\texttt{TPA}| + |\texttt{FNA}| + |\texttt{FPA}|}\\
	\texttt{AssA} &= \frac{1}{|\texttt{TP}|}\sum_{c\in \texttt{TP}}\texttt{Ass-IoU}\left(c\right)\\
	\\
	\mathcal{A} \left( c \right) &= \frac{|\texttt{TPA}\left(c\right)|}{|\texttt{TPA}\left(c\right)| + |\texttt{FNA}\left(c\right)| + |\texttt{FPA}\left( c \right) |}
	\quad
	\begin{cases}
		\texttt{TPA}\text{:  True Positive Associations}\\
		\texttt{FNA}\text{: False Negative Associations}\\
		\texttt{FPA}\text{: False Positive Associations}
	\end{cases}

The overall :math:`\texttt{HOTA}` is the integral over the :math:`\texttt{HOTA}_\alpha` for all values of :math:`\alpha`.

.. math::

	\texttt{HOTA} &= \int_0^1 \texttt{HOTA}_\alpha \, d\alpha\\

Within this code, the :math:`\texttt{HOTA}` is computed using the evaluation tools of the respective datasets.
For the DanceTrack dataset the DanceTrack internal `TrackEval <https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval>`_ package was used.
For the PoseTrack21 dataset, the `PoseTrack21 Evaluation Kit <https://github.com/anDoer/PoseTrack21/tree/main/eval/posetrack21>`_ was used.
All modules are (or at least can be) loaded as submodules while setting up the DGS framework.

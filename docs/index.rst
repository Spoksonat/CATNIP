CATNIP Documentation
====================

.. image:: _static/catnip_logo.png
   :align: center
   :width: 300px

Comprehensive Analysis Toolkit for Near-field Imaging and Phase-retrieval (**CATNIP**)
--------------------------------------------------------------------------------------

**CATNIP** is a software tool for rapidly generating simulated datasets based on wave-optics simulations, intended for testing multimodal retrieval algorithms in X-ray near-field imaging. It is developed and maintained by the `High-Energy Physics Research Group <https://hep.uniandes.edu.co/miembros/investigadores>`_ at Universidad de los Andes (Uniandes).

The frontend and backend are implemented in Python, and the graphical interface runs within the **CATNIP** virtual environment, whose installation is described later in this manual.

Features
--------

- Fast generation of simulated datasets for X-ray near-field imaging.
- Wave-optics simulations based on the projection approximation (valid for thin samples).
- Support for multiple imaging modalities:
  
  - Edge Illumination (EI) [Olivo2021]_
  - Structured Grating-based Imaging (SGBI) [Hipp2016]_
  - Sandpaper-based Imaging (SBI) [DiTrapani2024]_
  - Propagation-based Imaging (PBI) [Gradl2017]_

- Data generation for both 2D multimodal retrieval and CT reconstruction.
- Implementations of state-of-the-art phase retrieval algorithms:
  
  - Unified Modulated Pattern Analysis (UMPA) [DeMarco2023]_
  - Local Contrast Separation (LCS) [Quenot2021]_

- Modular Python codebase for easy extension and integration.
- Graphical user interface for simulation setup and visualization.

Installation
------------

See :doc:`installation` for instructions on setting up the **CATNIP** virtual environment and dependencies.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules

References
----------

.. [Olivo2021] Olivo, A., et al. "Edge-illumination X-ray phase-contrast imaging." (2021).
.. [Hipp2016] Hipp, A., et al. "Structured grating-based X-ray imaging." (2016).
.. [DiTrapani2024] Di Trapani, V., et al. "Sandpaper-based X-ray phase imaging." (2024).
.. [Gradl2017] Gradl, R., et al. "Propagation-based X-ray phase-contrast imaging." (2017).
.. [DeMarco2023] De Marco, F., et al. "Unified Modulated Pattern Analysis for phase retrieval." (2023).
.. [Quenot2021] Quenot, M., et al. "Local Contrast Separation for multimodal imaging." (2021).
.. [Li2017] Li, K., et al. "Multi-slice approach for X-ray phase imaging." (2017).

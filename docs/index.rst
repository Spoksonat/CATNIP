CATNIP Documentation
====================

.. image:: _static/Logo_CATNIP.pdf
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
  - Single Grating-Based Imaging (SGBI) [Hipp2016]_
  - Speckle-Based Imaging (SBI) [DiTrapani2024]_
  - Propagation-Based Imaging (PBI) [Gradl2017]_

- Data generation for both 2D multimodal retrieval and CT reconstruction.
- Implementations of state-of-the-art phase retrieval algorithms:
  
  - Unified Modulated Pattern Analysis (UMPA) [DeMarco2023]_
  - Low Coherence System (LCS) [Quenot2021]_

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

.. [Olivo2021] Alessandro Olivo. “Edge-illumination x-ray phase-contrast imaging”. In: Journal of Physics: Condensed Matter 33.36 (July 2021), p. 363002. DOI: 10.1088/1361-648X/ac0e6e. URL: https://dx.doi.org/10.1088/1361-648X/ac0e6e.
.. [Hipp2016] A. Hipp et al. “Single-grating interferometer for high-resolution phase-contrast imaging at synchrotron radiation sources”. In: Developments in X-Ray Tomography X. Ed. by Stuart R. Stock, Bert Müller, and Ge Wang. Vol. 9967. International Society for Optics and Photonics. SPIE, 2016, p. 996718. DOI: 10.1117/12.2237582. URL: https://doi.org/10.1117/12.2237582.
.. [DiTrapani2024] Vittorio Di Trapani et al. “Speckle-based imaging (SBI) applications with spectral photon counting detectors at the newly established OPTIMATO (OPTimal IMAging and TOmography) laboratory”. In: Journal of Instrumentation 19.01 (Jan. 2024), p. C01018. DOI: 10.1088/1748-0221/19/01/C01018. URL: https://dx.doi.org/10.1088/1748-0221/19/01/C01018.
.. [Gradl2017] Regine Gradl et al. “Propagation-based Phase-Contrast X-ray Imaging at a Compact Light Source”. In: Scientific Reports 7 (July 2017), p. 4908. DOI: 10.1038/s41598-017-04739-w.
.. [DeMarco2023] Fabio De Marco et al. “High-speed processing of X-ray wavefront marking data with the Unified Modulated Pattern Analysis (UMPA) model”. In: Opt. Express 31.1 (Jan. 2023), pp. 635–650. DOI: 10.1364/OE.474794. URL: https://opg.optica.org/oe/abstract.cfm?URI=oe-31-1-635.
.. [Quenot2021] Laurène Quénot et al. “Implicit tracking approach for X-ray phase-contrast imaging with a random mask and a conventional system”. In: Optica 8.11 (Nov. 2021), pp. 1412–1415. DOI: 10.1364/OPTICA.434954. URL: https://opg.optica.org/optica/abstract.cfm?URI=optica-8-11-1412.
.. [Li2017] Kenan Li, Michael Wojcik, and Chris Jacobsen. “Multislice does it all; calculating the performance of nanofocusing X-ray optics”. In: Opt. Express 25.3 (Feb. 2017), pp. 1831–1846. DOI: 10.1364/OE.25.001831. URL: https://opg.optica.org/oe/abstract.cfm?URI=oe-25-3-1831.

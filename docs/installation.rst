Installation
============

To install CATNIP, follow these steps:

1. Clone the repo:
   
   .. code-block:: bash

      git clone https://github.com/Spoksonat/CATNIP.git

2. Create the CATNIP virtual environment:

   .. code-block:: bash

      python3 -m venv CATNIP_env

3. Activate the CATNIP virtual environment:

For Linux/MacOs:

   .. code-block:: bash

      source CATNIP_env/bin/activate

For Windows:

   .. code-block:: bash

      CATNIP_env\Scripts\activate.bat

4. Install the required dependencies:

   .. code-block:: bash

      pip install -r CATNIP/requirements.txt

5. Verify the installed packages:

   .. code-block:: bash

      pip list

6. Deactivate virtual environment:

   .. code-block:: bash

      deactivate

Note: If your default Python version is older than 3.12.2, and you have installed a compatible version separately, replace the command python3 with the full path to the correct Python binary, e.g., /path/to/python3.12. To locate the path, you can use:

   .. code-block:: bash

      which python3.12



   


# CATNIP (Linux, Windows and MacOS)/ Under development

CATNIP stands for Comprehensive Analysis Toolkit for Near-field Imaging and Phase-retrieval.

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About the Project

Comprehensive Analysis Toolkit for Near-field Imaging and Phase-retrieval (CATNIP) is a software tool for rapidly generating simulated datasets based on wave-optics simulations, intended for testing multimodal retrieval algorithms in X-ray near-field imaging. It is developed and maintained by the High-Energy Physics Research Group at Universidad de los Andes (Uniandes). The frontend and backend are implemented in Python, and the graphical interface runs within the CATNIP virtual environment, whose installation is described in the manual.

## Getting Started

To get a local copy up and running, read the manual or follow these simple steps.

### Installation

Step-by-step guide on how to get the development environment running.

1. Clone the repo:
   ```sh
   git clone https://github.com/Spoksonat/CATNIP.git
   ```
2. Create the CATNIP virtual environment:
   ```sh
   python3 -m venv CATNIP_env
   ```
3. Activate the CATNIP virtual environment:

   For Linux/MacOs:
   ```sh
   source CATNIP_env/bin/activate
   ```

   For Windows:
   ```sh
   CATNIP_env\Scripts\activate.bat
   ```
4. Install the required dependencies:
   ```sh
   pip install -r CATNIP/requirements.txt
   ```
5. Verify the installed packages:
   ```sh
   pip list
   ```
6. Deactivate virtual environment:
   ```sh
   deactivate
   ```
Note: If your default Python version is older than 3.12.2, and you have
installed a compatible version separately, replace the command python3
with the full path to the correct Python binary, e.g., /path/to/python3.12.
To locate the path, you can use:

```sh
   which python3.12
```


## Usage

To use this project, read the manual or follow the instructions below.

1. Activate the CATNIP virtual environment:
   
   For Linux/MacOs:
   ```sh
   source CATNIP_env/bin/activate
   ```

   For Windows:
   ```sh
   CATNIP_env\Scripts\activate.bat
   ```
2. Navigate to the CATNIP main directory:
   ```sh
   cd CATNIP
   ```
3. Run the main file:
   ```sh
   python main.py
   ```

If you want to know more details about using this software, you can refer to the [Manual](CATNIP_Manual.pdf).

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have a suggestion for improving this project, please fork the repository and create a pull request. You can also open an issue with the tag "enhancement". Don't forget to give the project a star! Thank you!

1. Fork the Project.
2. Create your Feature Branch (\`git checkout -b feature/AmazingFeature\`).
3. Commit your Changes (\`git commit -m 'Add some AmazingFeature'\`).
4. Push to the Branch (\`git push origin feature/AmazingFeature\`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See [MIT License](https://github.com/Spoksonat/CATNIP/blob/main/LICENCE.txt) for more information.

## Contact

Manuel Fernando Sánchez Alarcón - mf.sanchez17@uniandes.edu.co

Project Link: [CATNIP](https://github.com/Spoksonat/CATNIP)

## Acknowledgments

- [Pierre Thibault](https://github.com/pierrethibault), for his UMPA implementation, which is included in this software.

**Artificial Bartender**

*Installation instructions:*

1. (Linux only) install tkinter: `sudo apt-get install python3-tk`
2. Create virtual environment with python3: `virtualenv venv --python=python3`
3. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux)
4. Install requirements: `pip install -r requirements.txt`
5. (Windows only) install pywin32: `pip install pywin32`
6. Run setup with `python setup.py` to download age_gender_models


*Launch*

`./main.py`


***
*3rd Party Modules*

Pre-trained neural network models are taken from: https://talhassner.github.io/home/publication/2015_CVPR

Haar Cascades are taken from OpenCV: https://pypi.org/project/opencv-python/3.4.2.17

Bitwise Font by Digital Graphic Labs: https://www.1001fonts.com/bitwise-font.html
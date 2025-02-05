# augmented-reality

Augmented reality card based application with `Python`, `numpy` and `OpenCV`.

## Usage

* Place the image of the surface to be tracked inside the `reference` folder.
* On line 16 of `ar_main.py` replace the first argument to the class `AR_3D` with the name of the image you just copied inside the `reference` folder (*note:* at the moment the extension is assumed to be `.jpg`).
* The 3D object to render is chosen as user input argument (see next section), by default it is rendered the first `.obj` file in the `models` folder. To change the size of the rendered model change the scale parameter (number `3`) in line 164 of `src/aaugmented_reality_3D.py` by a suitable number. This might require some trial and error.
* Open a terminal session inside the project folder and run `python ar_main.py`

## Command line arguments

Check the input argument help, type:

    python ar_main.py -h

## Troubleshooting

**If you get the message**:

    Unable to capture video

printed to your terminal, the most likely cause is that your OpenCV installation has been compiled without FFMPEG support. Pre-built OpenCV packages such as the ones downloaded via pip are not compiled with FFMPEG support, which means that you have to build it manually.

## Framework

The code as been tested in `python 3.7.4` but I suppose any `3.x` version should work. The only required package should be: *cv2* (openCV), *numpy*.  
To check the python notebook install [jupiter](https://jupyter.org/install) or [jupiter lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.h).  
Another way is to install all the required package via:

    pip install -r requirements.txt

which lists *all* the packages, also those required and installed together with *cv2* and *numpy* but also some additional packages (e.g. some linting packages) that maybe you don't want.

**Remark:** check this [guide](environment_setup.md) to understand how I created a dedicated virtual environment for this project.

## Explanation

Check the python notebook in the `notebook` folder or read the blog entries for an in-depth explanation of the logic behind the code:

* [Part 1](https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/)
* [Part 2](https://bitesofcode.wordpress.com/2018/09/16/augmented-reality-with-python-and-opencv-part-2/)

## Results

* [Mix](https://www.youtube.com/watch?v=YVJSFcUbIoU)
* [Fox](https://www.youtube.com/watch?v=V13VE6UJ-1g)
* [Ship](https://www.youtube.com/watch?v=VDwfW75f3Xo)
* [Rat](https://www.youtube.com/watch?v=Bb7pYthMM64)
* [Cow](https://www.youtube.com/watch?v=f0fNzXP3ku8)
* [Fox II](https://www.youtube.com/watch?v=_fozNTdql6U)
* [Fox III](https://www.youtube.com/watch?v=FGKkIr_IIy4)

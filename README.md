# ml_learn

A GitHub repo for learning ML.

Most of the examples here are based on my learning from the book:  "Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow" by Aurelien Geron.

# Installation

First make sure you are running Python 3.6 or above.

```
python --version
```

Next, it is recommended that you create a virtual environment:

```
cd ml_learn
python -m venv .venv
```

To activate the virtual environment, run:

```
source .venv/bin/activate
```

You are now ready to install the package:

```
python setup.py develop
```

# Command Line Usage

To run the ```ml``` command line tool, make sure you have first activated your virtual environment:

```
source .venv/bin/activate
```

Then, type:

```
ml
```

# License

MIT License

Copyright (c) ecerami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
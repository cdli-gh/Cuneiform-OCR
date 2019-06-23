# Synthetic Dataset generation

## Pre-requisites

All the pre-requisite libraries can be found in requirements.txt.

Download the B/W sign and line images from here: https://drive.google.com/open?id=1g4JeaJrmQ8K_fh_TOYeAyd6s6grqH2Dr

## Usage

To generate gradient augmented signs:

```
python sign_gradient_generation.py './input/signs/'
```

After the gradient augmented signs are generated, run ```python sign_placement.py``` to generate the templates with lines and cuneiform signs placed. 

Dimensions of the template(width/height) can be adjusted in the code.


## Expected Output

<img src="https://i.imgur.com/EVLYMPA.png" width="200" height="245"> <img src="https://i.imgur.com/d8vurLg.png" width="200" height="245">

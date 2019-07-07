# Synthetic Dataset generation

## Pre-requisites

All the pre-requisite libraries can be found in requirements.txt.

Download dataset for synthetic image generation: https://drive.google.com/open?id=1g4JeaJrmQ8K_fh_TOYeAyd6s6grqH2Dr

The above includes 78 cuneiform signs with augmented variants, line images.


## Usage

To generate gradient augmented signs:

```
python sign_gradient_generation.py './input/signs/'
```

After the gradient augmented signs are generated, run ```python sign_placement.py``` to generate the templates with lines and cuneiform signs placed. 

Dimensions of the template(width/height) can be adjusted in the code.


## Expected Output

Final generate templates can be downloaded here:

<img src="https://i.imgur.com/TByxMc7.png" width="200" height="245"> <img src="https://i.imgur.com/BWK06iG.png" width="204.6" height="302"> <img src="https://i.imgur.com/qaDqpMf.png" width="165" height="173"> <img src="https://i.imgur.com/iPdX9Hn.png" width="217" height="254">

<img src="https://i.imgur.com/X9xUYAp.png" width="200" height="245"> <img src="https://i.imgur.com/snBNcLM.png" width="204.6" height="302"> <img src="https://i.imgur.com/wTzzexK.png" width="165" height="173"> <img src="https://i.imgur.com/TTm4lx6.png" width="217" height="254">

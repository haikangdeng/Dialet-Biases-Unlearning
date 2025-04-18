# Exploiting Cultural Biases via Homoglyphs in Text-to-Image Synthesis

  <center>
  <img src="images/concept.jpg" alt="Exploiting Cultural Biases via Homoglyphs in Text-to-Image Synthesis"  height=300>
  </center>

> **Abstract:**
> *Models for text-to-image synthesis, such as DALL-E~2 and Stable Diffusion, have recently drawn a lot of interest from academia and the general public. These models are capable of producing high-quality images that depict a variety of concepts and styles when conditioned on textual descriptions. However, these models adopt cultural characteristics associated with specific Unicode scripts from their vast amount of training data, which may not be immediately apparent. We show that by simply inserting single non-Latin characters in a textual description, common models reflect cultural stereotypes and biases in their generated images. We analyze this behavior both qualitatively and quantitatively and identify a model's text encoder as the root cause of the phenomenon. Additionally, malicious users or service providers may try to intentionally bias the image generation to create racist stereotypes by replacing Latin characters with similarly-looking characters from non-Latin scripts, so-called homoglyphs. To mitigate such unnoticed script attacks, we propose a novel homoglyph unlearning method to fine-tune a text encoder, making it robust against homoglyph manipulations*  
[Full Paper (PDF)](https://arxiv.org/pdf/2209.08891.pdf)


## Setup Docker Container
The easiest way to perform the attacks is to run the code in a Docker container. To build the Docker image run the following script:

```bash
docker build -t exploiting_homoglyphs  .
```

To create and start a Docker container run the following command from the project's root:

```bash
docker run --rm --shm-size 16G --name my_container --gpus '"device=0"' -v $(pwd):/workspace -it exploiting_homoglyphs bash
```

## Generate Stable Diffusion Images
To reproduce our qualitative results from the paper and generate (biased) images, use the scripts ```generate_stable_diffusion_images.py``` and ```generate_stable_diffusion_images_embedding_diff.py```. To load the Stable Diffusion model from Hugging Face, you need to provide the token from a user account. A token can be created at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Save your token in the ```HF_TOKEN``` variable in the scripts. Further, specify the injected homoglyphs in the ```HOMOGLYPHS``` tuple. For generating images with homoglyphs in the textual description, specify the prompt and the character to be replaced with homoglyphs in line 32 of ```generate_stable_diffusion_images.py```. For inducing the biases in the embedding space, state the text prompt and the Latin character to compute the difference from in the ```PROMPTS``` variable in ```generate_stable_diffusion_images_embedding_diff.py```.

Then run the scripts with the following commands:
```bash
python generate_stable_diffusion_images.py
python generate_stable_diffusion_images_embedding_diff.py
```
  <center>
  <img src="images/character_bias.jpg" alt="Inducing biases in the embedding space"  height=300>
  </center>

## Compute WEAT Scores
To repeat the WEAT test from Table 1 in our paper, run ```python compute_weat.py```. To compute the test on the multilingual clip encoder (M-CLIP), set the flag ```MULTILINGUAL``` in line 22 of the file to ```True```.

## Compute Relative Bias
To compute the relative bias as in Figure 6 in the paper, run ```python compute_relative_bias.py```. To change the investigated characters or prompt sets, adjust the variables ```HOMOGLYPHS``` and ```TEMPLATES```, respectively. To replace the standard text encoder with a another one from WANDB, provide the run path for a trained encoder in the variable ```ENCODER_RUN_PATH```. This option is particularly designed to evaluate the homoglyph unlearning approach.
  <center>
  <img src="images/relative_bias_concept.jpg" alt="Relative bias concept"  height=150>
  </center>


## Homoglyph Unlearning
To perform the homoglyph unlearning approach, run ```python homoglyph_unlearning.py -c=configs/homoglyph_unlearning.yaml```. If the process takes too much GPU memory, try to reduce the value of the parameters ```poisoned_samples_per_step``` and ```clean_batch_size``` in the configuration file ```configs/homoglyph_unlearning.yaml```. Generally, all training-related hyperparameters can be set in the configuration file. Our code is based on backdoor attacks against text-to-image synthesis models. For more information, see [https://github.com/LukasStruppek/Rickrolling-the-Artist](https://github.com/LukasStruppek/Rickrolling-the-Artist) and the corresponding paper [Rickrolling the Artist: Injecting Invisible Backdoors into Text-Guided Image Generation Models](arxiv.org/abs/2211.02408).
  <center>
  <img src="images/unlearning_concept.jpg" alt="Homoglyph unlearning concept"  height=240>
  </center>
  <center>
  <img src="images/unlearning_samples.jpg" alt="Homoglyph unlearning results"  height=240>
  </center>

# Citation
If you build upon our work, please don't forget to cite us.
```
@article{struppek22homoglyphs,
  author = {Struppek, Lukas and Hintersdorf, Dominik and Friedrich, Felix and Brack, Manuel and Schramowski, Patrick and Kersting, Kristian},
  title = {Exploiting Cultural Biases via Homoglyphs in Text-to-Image Synthesis},
  journal = {Journal of Artificial Intelligence Research (JAIR)},
  volume = {78},
  year = {2023},
  pages = {1017--1068},
}
```


# Packages and Repositories
Some of our analyses rely on other repos and pre-trained models. We want to thank the authors for making their code and models publicly available. For license details, refer to the corresponding files in our repo. For more details on the specific functionality, please visit the corresponding repos.

- CLIP: https://github.com/openai/CLIP
- Open-CLIP: https://github.com/mlfoundations/open_clip
- Multilingual CLIP: https://github.com/FreddeFrallan/Multilingual-CLIP
- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- DALL-E 2 Python API: https://github.com/ezzcodeezzlife/dalle2-in-python
- WEAT Implementation: https://github.com/ryansteed/weat

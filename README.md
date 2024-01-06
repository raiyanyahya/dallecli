[![Actions Status](https://github.com/raiyanyahya/dallecli/workflows/Build%20Test/badge.svg)](https://github.com/raiyanyahya/dallecli/actions) [![Actions Status](https://github.com/raiyanyahya/dallecli/workflows/Package%20Release/badge.svg)](https://github.com/raiyanyahya/dallecli/actions) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=raiyanyahya_dallecli&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=raiyanyahya_dallecli)[![CodeQL](https://github.com/raiyanyahya/dallecli/workflows/CodeQL/badge.svg)](https://github.com/raiyanyahya/dallecli/actions?query=workflow%3ACodeQL) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/851417bc6ec8405ba244438fe31dae55)](https://www.codacy.com/gh/raiyanyahya/dallecli/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=raiyanyahya/dallecli&amp;utm_campaign=Badge_Grade) [![](https://img.shields.io/badge/python-3.6+-blue.svg)] 
[![PyPI version](https://badge.fury.io/py/dallecli.png)](https://badge.fury.io/py/dallecli)[![PyPI download month](https://img.shields.io/pypi/dm/dallecli.svg)](https://pypi.python.org/pypi/dallecli/)

# DalleCli 💠

`dallecli` is a community-maintained cli designed to provide users with the ability to generate, edit and filter images using the DALL-E 3 API provided by OpenAI, all from the command line.

The tool provides three main commands, generate, edit, and filter.

The `generate` command allows the user to generate an image using a prompt, with the option to apply a filter, specify the size, and choose the number of times to generate the image. The generated image can also be saved to a specified file path.

The `edit` command provides the ability to edit an existing image by adjusting the brightness, contrast, and sharpness of the image. The edited image is then saved to a new file.

The `filter` command allows users to apply various filters and effects to an existing image. The user can select from a range of different filters, and the filtered image can be saved to a specified file path.

This is ideal for developers, designers, and anyone who wants to quickly generate and manipulate images without the need for a full-fledged image editing software. I hope you find it useful.

## Note

The image generations endpoint allows you to create an original image given a text prompt. When using DALL·E 3, images can have a size of 1024x1024, 1024x1792 or 1792x1024 pixels.

By default, images are generated at standard quality, but when using DALL·E 3 you can set quality: "hd" for enhanced detail. Square, standard quality images are the fastest to generate.

You can request 1 image at a time with DALL·E 3 (request more by making parallel requests) 

Read more here https://platform.openai.com/docs/guides/images/introduction

## Configuration

The cli requires you to have an api token to query the OpenAI's api. You can read about and get it here https://platform.openai.com/account/api-keys.

## Installation

Install the dallecli python package directly from pypi. 

```console
  pip install dallecli
```
I would recommend using pipx instead of pip to install cli applications on you machine.

## Usage

```console
Usage: dallecli [OPTIONS] COMMAND [ARGS]...

  💠 Use the Dall.E 3 api to generate, edit & filter images from the cmd line.

Options:
  --help  Show this message and exit.

Commands:
  edit      🎴 Change the brightness, contrast and sharpness of an image.
  filter    🦄 Apply filters and effects to an image.
  generate  🌸 Generate an image from the OpenAI Dalle api.
  update    🔐 Update the OpenAI API key.
```

### Commands and Options

**```generate```**
```console
Usage: dallecli generate [OPTIONS]

  🌸 Generate an image from the OpenAI Dalle api

Options:
  --prompt TEXT                   💬 The prompt to generate the image from.
  --size TEXT                     📐 The size of the generated image (default 1024x1024).
  --filter                        🎨 Apply a filter to the generated image.
  --iterations INTEGER            🔄 The number of times to generate the image.
  --save-path FILE                💾 Save the generated image to the specifiedfile path.
  --hide FLAG                     🖱️ Do not open the image after generation.
  --quality TEXT                  👌 The quality of the image ( "standard" or "hd" ).
  --model TEXT                    🦾 The OpenAI model to use when generating images ( use dall-e-2 for older model)
  --help                          Show this message and exit.
```

**```edit```**
```console
Usage: dallecli edit [OPTIONS] IMAGE_PATH

  🎴 Change the brightness, contrast and sharpness of an image.

Options:
  --brightness FLOAT  💡 The brightness of the image
  --contrast FLOAT    🌈 The contrast of the image
  --sharpness FLOAT   🔪 The sharpness of the image
  --help              Show this message and exit.
```

**```filter```**
```console
Usage: dallecli filter [OPTIONS] IMAGE_PATH

  🦄 Apply filters and effects to an image.

Options:
  --filter [grayscale|sepia|blur|contour|detail|edge_enhance|edge_enhance_more|emboss|find_edges|sharpen|smooth|smooth_more|outline|posterize|solarize|invert|flip]
                                  🎨 The filter to apply to the image
  --save-path PATH                The directory to save the filtered image.
                                  Defaults to the current directory.
  --help                          Show this message and exit.
```

**```update```**
```console
Usage: dallecli update [OPTIONS]

  🔐 Update the OpenAI API key.

Options:
  --env   ♻️ Get the api key from the OPENAI_API_KEY env variable
  --help  Show this message and exit.
```

Please feel to create issues or request for features. There will be many features added to the cli.

import click
from os import path, makedirs, getenv, environ
from io import BytesIO
from openai import OpenAI, AuthenticationError
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import requests
from rich.progress import Console
import json

console = Console()
client = OpenAI(api_key="")


def configure_openai():
    config_file = path.expanduser("~/.openai/config.json")
    if not path.exists(config_file):
        makedirs(path.dirname(config_file), exist_ok=True)
        api_key = input("🔑 Enter your OpenAI API key: ")
        with open(config_file, "w", encoding="UTF-8") as f:
            f.write(f'{{"api_key": "{api_key}"}}')
    else:
        with open(config_file, "r", encoding="UTF-8") as f:
            api_key = json.load(f).get("api_key")
        if not api_key:
            api_key = input("🔑 Enter your OpenAI API key: ")
            with open(config_file, "w", encoding="UTF-8") as f:
                f.write(f'{{"api_key": "{api_key}"}}')
    client.api_key = api_key or getenv("OPENAI_API_KEY")


def generate_image(
    prompt, size, filter, iterations, hide, quality, model, save_path=None
):
    try:
        with console.status("Generating image...", spinner="dots8Bit"):
            for _ in range(iterations):
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    **({"filter": filter} if filter else {}),
                )
                image_data = requests.get(response.data[0].url, timeout=300).content
                image = Image.open(BytesIO(image_data))
                if not hide:
                    image.show()
                if save_path is not None:
                    if not path.exists(path.dirname(save_path)):
                        makedirs(path.dirname(save_path))
                    image.save(save_path)
    except AuthenticationError:
        print("🔒 Authentication Failed. Try with a fresh API key.")
    except Exception as e:
        print("❌ Failed to generate image. Please try again with a different prompt.")
        print(f"Error: {e}")


def edit_image(image, brightness=None, contrast=None, sharpness=None):
    if brightness is not None:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast is not None:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if sharpness is not None:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image


def apply_filter_choices(image, filter_name):
    filters = {
        "grayscale": lambda i: i.convert("L"),
        "sepia": lambda i: i.convert("RGB")
        .convert("L")
        .convert(
            "RGB",
            matrix=(
                0.393,
                0.769,
                0.189,
                0.349,
                0.686,
                0.168,
                0.272,
                0.534,
                0.131,
            ),
        ),
        "blur": lambda i: i.filter(ImageFilter.BLUR),
        "contour": lambda i: i.filter(ImageFilter.CONTOUR),
        "detail": lambda i: i.filter(ImageFilter.DETAIL),
        "edge_enhance": lambda i: i.filter(ImageFilter.EDGE_ENHANCE),
        "edge_enhance_more": lambda i: i.filter(ImageFilter.EDGE_ENHANCE_MORE),
        "emboss": lambda i: i.filter(ImageFilter.EMBOSS),
        "find_edges": lambda i: i.filter(ImageFilter.FIND_EDGES),
        "sharpen": lambda i: i.filter(ImageFilter.SHARPEN),
        "smooth": lambda i: i.filter(ImageFilter.SMOOTH),
        "smooth_more": lambda i: i.filter(ImageFilter.SMOOTH_MORE),
        "outline": lambda i: i.filter(ImageFilter.CONTOUR).filter(
            ImageFilter.SMOOTH_MORE
        ),
        "posterize": lambda i: i.convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=10
        ),
        "solarize": lambda i: ImageOps.solarize(i, threshold=128),
        "invert": lambda i: ImageOps.invert(i),
        "flip": lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "rotate_90": lambda i: i.transpose(Image.Transpose.ROTATE_90),
        "rotate_180": lambda i: i.transpose(Image.Transpose.ROTATE_180),
        "rotate_270": lambda i: i.transpose(Image.Transpose.ROTATE_270),
    }
    if filter_name in filters:
        return filters[filter_name]
    return image


@click.group()
@click.version_option(version="2.2.0")
def cli():
    """💠 Use the Dall.E 3 api to generate, edit & filter images from the cmd line."""


@cli.command("generate")
@click.option(
    "--prompt",
    default="Suprise me",
    prompt=True,
    help="💬 The prompt to generate the image from.",
)
@click.option("--size", default="1024x1024", help="📐 The size of the generated image.")
@click.option(
    "--filter",
    type=click.Choice(
        [
            "grayscale",
            "sepia",
            "blur",
            "contour",
            "detail",
            "edge_enhance",
            "edge_enhance_more",
            "emboss",
            "find_edges",
            "sharpen",
            "smooth",
            "smooth_more",
            "outline",
            "posterize",
            "solarize",
            "invert",
            "flip",
        ]
    ),
    help="🎨 Apply a filter to the generated image.",
)
@click.option(
    "--iterations", default=1, help="🔄 The number of times to generate the image"
)
@click.option(
    "--save-path",
    type=click.Path(dir_okay=False, writable=True),
    help="💾 Save the generated image to the specified file path",
    default="images/output.png",
)
@click.option("--hide", is_flag=True, help="🖱️ Do not open the image after generation")
@click.option("--quality", default="standard", help="👌 The quality of the image")
@click.option(
    "--model",
    default="dall-e-3",
    help="🦾 The OpenAI model to use when generating images",
)
def generate(prompt, size, filter, iterations, save_path, hide, quality, model):
    """🌸 Generate an image from the OpenAI Dalle api"""
    configure_openai()
    generate_image(prompt, size, filter, iterations, hide, quality, model, save_path)


@cli.command("edit")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--brightness", type=float, help="💡 The brightness of the image")
@click.option("--contrast", type=float, help="🌈 The contrast of the image")
@click.option("--sharpness", type=float, help="🔪 The sharpness of the image")
def edit(image_path, brightness, contrast, sharpness):
    """🎴 Change the brightness, contrast and sharpness of an image."""

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image file: {e}")
        return

    edited_image = edit_image(image, brightness, contrast, sharpness)

    output_file_path = get_output_edit_image_path(image_path)

    try:
        edited_image.save(output_file_path)
        print(f"✅ Image saved to {output_file_path}")
    except Exception as e:
        print(f" ❌ Error saving image file: {e}")
        return


@cli.command("update")
@click.option(
    "--env",
    is_flag=True,
    help="♻️ Get the api key from the OPENAI_API_KEY env variable",
)
def update_key(env):
    """🔐 Update the OpenAI API key."""
    config_file = path.expanduser("~/.openai/config.json")
    if not path.exists(config_file):
        makedirs(path.dirname(config_file), exist_ok=True)
    api_key = ""
    if env:
        try:
            api_key = environ["OPENAI_API_KEY"]
        except KeyError:
            print("🛑 $OPENAI_API_KEY variable not set.")
            return
    else:
        api_key = input("Enter your OpenAI API key: ")
    with open(config_file, "w", encoding="UTF-8") as f:
        f.write(f'{{"api_key": "{api_key}"}}')
    print("API key updated successfully!")


def get_output_edit_image_path(input_file_path):
    output_file_path = path.splitext(input_file_path)[0] + "_edited.png"
    return output_file_path


@cli.command("filter")
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--filter",
    type=click.Choice(
        [
            "grayscale",
            "sepia",
            "blur",
            "contour",
            "detail",
            "edge_enhance",
            "edge_enhance_more",
            "emboss",
            "find_edges",
            "sharpen",
            "smooth",
            "smooth_more",
            "outline",
            "posterize",
            "solarize",
            "invert",
            "flip",
        ]
    ),
    help="🎨 The filter to apply to the image",
)
@click.option(
    "--save-path",
    type=click.Path(),
    default=".",
    help="The directory to save the filtered image. Defaults to the current directory.",
)
def apply_filter(image_path, filter, save_path):
    """🦄 Apply filters and effects to an image."""
    image = Image.open(image_path)
    filtered_image_func = apply_filter_choices(image, filter)
    filtered_image = filtered_image_func(image)
    filtered_image_path = get_output_image_path(image_path, filter, save_path)
    if path.exists(filtered_image_path):
        filtered_image_path = get_unique_filename(filtered_image_path)
    filtered_image.save(filtered_image_path)
    click.echo(f"✅ Filtered image saved to {filtered_image_path}")


def get_output_image_path(image_path, prefix, save_path):
    _, image_filename = path.split(image_path)
    image_name, image_ext = path.splitext(image_filename)
    filtered_image_name = f"{prefix}_{image_name}{image_ext}"
    return path.join(save_path, filtered_image_name)


def get_unique_filename(filename):
    name, ext = path.splitext(filename)
    i = 1
    while path.exists(filename):
        filename = f"{name}_{i}{ext}"
        i += 1
    return filename

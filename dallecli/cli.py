#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import pickle
import signal
from datetime import datetime
from os import path, makedirs, getenv
from sys import path as sys_path, stderr
from io import BytesIO
from pickle import Pickler

pickle.HIGHEST_PROTOCOL = 4

# popen
from subprocess import Popen, PIPE
import typing
from typing import Any, Callable, Optional, NoReturn
import webbrowser

try:
    import click
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ExifTags
    from PIL.Image import Exif
    import openai
    from openai import OpenAI, AuthenticationError
    from openai.types import ImagesResponse
    import requests
    from rich.console import Console
except ImportError as e:
    print(f"❌ Error importing module: {e.path}")
    print("📦 Please install the required dependencies and try again.")
    exit(1)

console = Console()
root_logger = logging.getLogger(__name__)
root_logger.setLevel(logging.INFO)
logger = root_logger.getChild("cli")
logger.setLevel(logging.DEBUG)


def configure_openai():
    openai_client = OpenAI()
    config_file = path.expanduser("~/.openai/config.json")
    api_key = None
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
    if not api_key:
        api_key = getenv("OPENAI_API_KEY")
        if not api_key:
            print(
                "❌ No API key found. Please run `dallecli update` to update your API key."
            )
            exit(1)
    openai_client.api_key = api_key
    return openai_client


def _has_kitty() -> bool:
    """Check if kitty is installed."""
    logger.debug("Checking if kitty is installed…")
    for p in sys_path:
        kitty_path = path.join(p, "kitty")
        logger.debug(f"Checking {kitty_path}…")
        if not path.exists(kitty_path):
            continue
        kitty_path = os.path.realpath(kitty_path)
        if (
            not path.exists(kitty_path)
            or not path.isfile(kitty_path)
            or not os.access(kitty_path, os.X_OK)
        ):
            continue
        return True
    return False


def generate_image(
    openai_client: OpenAI,
    prompt,
    size,
    filter,
    iterations,
    hide,
    model,
    quality,
    skip_exif,
    save_path=None,
) -> bool:
    global interrupted
    interrupted = False
    def _api_tx() -> tuple[ImagesResponse | None, str | None] | NoReturn:
        """Send the prompt to the OpenAI API and return the response."""
        logger.debug(f"📡 Sending prompt to OpenAI API…")

        response = None

        try:
            response = openai_client.images.generate(
                prompt=prompt,
                size=size,
                model=model,
                response_format="url",
                quality=quality,
            )
        except AuthenticationError:
            print("\n🔒 Authentication failed. Check your API key and try again.")
        except openai.BadRequestError as e:
            print(f"\n❌ Generation failed, prompt triggered their filters.")
            if e.code != "content_policy_violation":
                print(f"❓ Exception information: {e}")
        except Exception as e:
            print(f"\n❌ Generation failed.")
            print(f"❓ Exception information: {e}")
            return None, None

        assert response is not None

        print(file=stderr)
        logger.info(f"Response: {response}")
        data = response.data[0]
        url = data.url
        if url is None:
            data = response.data[0]
            return None, None
        else:
            url = data.url
        print(f"✅ Image generated!")
        return response, url

    def _get_image_from_url(url: str) -> Image.Image:
        """Download the image from the given URL and return it."""
        print(f"⏳ Downloading image from {url}…")
        try:
            response = requests.get(url, timeout=300 * 60)
            response.raise_for_status()
            image_data = response.content
            print(f"✅ Image downloaded. Size: {len(image_data)} bytes")
            image = Image.open(BytesIO(image_data))
        except requests.exceptions.ReadTimeout:
            print(f"❌ Downloading image from {url} timed out.")
            raise
        except requests.exceptions.HTTPError as e:
            print(f"❌ Failed to download image from {url}.")
            print(f"❓ Exception information: {e}")
            raise

        return image

    def _append_prompt_to_image(response, image, prompt) -> tuple[Image.Image, Exif]:
        def _format_exif(image: Image.Image) -> Exif:
            """Format the EXIF metadata."""
            exif = Image.Exif()
            response_data = response.data[0]
            if response_data.revised_prompt is not None:
                revised_prompt = response_data.revised_prompt
            else:
                revised_prompt = prompt
            exif.update(
                {
                    ExifTags.Base.XPComment: pickle_io.getvalue(),
                    ExifTags.Base.XPAuthor: f"{model} via dallecli",
                    ExifTags.Base.XPKeywords: prompt,
                    ExifTags.Base.XPSubject: "Generated image",
                    ExifTags.Base.XPTitle: revised_prompt,
                }
            )
            exif.update(image.getexif())
            return exif

        """Append the prompt to the image as EXIF metadata."""
        logger.debug("⏳ Appending prompt to image…")
        pickle_io = BytesIO()
        pickler = Pickler(pickle_io)
        pickler.dump(response)
        image = image.copy()
        exif = _format_exif(image)
        image.info["exif"] = exif.tobytes()

        logger.debug("⌛ Prompt appended to image.")
        return image, exif

    def _apply_filter_to_image(image, filters: list[str]) -> Image.Image:
        """Apply the specified filters to the image."""
        for filter_ in filters:
            image = apply_filter_choices(image, filter_)
        return image

    def _mangle_save_path(save_path: str|None) -> str:
        """Mangle the save path to include the prompt."""
        if save_path is None:
            save_path = "images/output.png"
        save_path = path.expanduser(save_path)
        save_path = path.abspath(save_path)
        save_path = get_unique_filename(save_path)
        dirname = path.dirname(save_path)
        if not path.exists(dirname) and dirname != "":
            makedirs(path.dirname(save_path))

        return save_path

    def _do_fs_io(image, save_path):
        """Save the image to the filesystem."""
        logger.debug(f"⏳ Saving image to {save_path}…")
        if skip_exif:
            image.save(save_path)
        else:
            image, exif = _append_prompt_to_image(response, image, prompt)
            assert "exif" in image.info and type(exif) is Exif
            image.save(save_path, exif=exif)
        logger.debug(f"✅ Image saved to {save_path}")
        return image, save_path

    def _check_image_size(image, size):
        ih, iw = size.split("x")
        ih, iw = int(ih), int(iw)
        nih, niw = image.size
        if nih != ih or niw != iw:
            e = ValueError(
                f"Image size mismatch. Expected {size}, got {image.height}x{image.width}."
            )
            print(f"❌ {e}")
            return False
        return True

    def _show_image(save_path):
        """Show the image in the terminal."""
        if not _has_kitty():
            webbrowser.open(save_path)
        else:
            image = Image.open(save_path)
            if "exif" in image.info:
                exif = image.getexif()
                if ExifTags.Base.XPTitle in exif:
                    revised_prompt = exif[ExifTags.Base.XPTitle]
                    revised_prompt = click.style(revised_prompt, italic=True)
                    msg_title = f"“Revised” prompt"
                    msg_title = click.style(msg_title, underline=True)
                    revised_prompt = click.wrap_text(
                        revised_prompt, width=50, initial_indent="    "
                    )
                    revised_prompt = click.style(revised_prompt, fg=(128, 128, 128))
                    msg = f"\n📝 {msg_title}:\n{revised_prompt}"
                    print(msg, file=stderr)
            iw, ih = image.size
            niw = 512
            nih = int((niw / iw) * ih)
            image = image.resize((niw, nih))
            _kitty = "kitty +kitten icat --align=left --stdin yes".split(" ")
            kitty = Popen(
                _kitty,
                stdin=PIPE,
            )
            print(file=stderr)
            with BytesIO() as io:
                image.save(io, format="PNG")
                assert kitty.stdin is not None
                kitty.stdin.write(io.getvalue())
                kitty.communicate()

    retval: list[bool] = []
    interrupted = False
    old_save_path = save_path
    now = datetime.now()

    for sig in (signal.SIGINT, signal.SIGTERM,
                signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(sig, signal_handler)

    for i in range(1, iterations + 1):
        if interrupted:
            print(f"🛑 Will not generate remaining {iterations - i + 1} images due to interruption.")
            break
        if iterations > 1:
            msgextra = f" №{i}/{iterations}…"
        else:
            msgextra = "…"
        make_msg = lambda msg: f"⏳ {msg}{msgextra}"
        response = None
        url = None
        image = None
        new_save_path = None
        try:
            msg = make_msg("Generating image")
            with console.status(msg, spinner="dots8Bit"):
                response, url = _api_tx()
                if response is None or url is None:
                    retval.append(False)
                    continue
            save_path = _mangle_save_path(save_path)
            msg = make_msg("Downloading image")
            with console.status(msg, spinner="dots8Bit"):
                image = _get_image_from_url(url)
            msg = make_msg("Applying filters")
            with console.status(msg, spinner="dots8Bit"):
                image = _apply_filter_to_image(image, filter)
            msg = make_msg("Saving image to filesystem")
            with console.status(msg, spinner="dots8Bit"):
                image, save_path = _do_fs_io(image, save_path)
            new_save_path = save_path
            save_path = old_save_path
            if not _check_image_size(image, size):
                retval.append(False)
                continue
            msg = make_msg("Showing image")
            with console.status(msg, spinner="dots8Bit"):
                if not hide:
                    _show_image(new_save_path)
            print(f"✅ Done! Image saved to {new_save_path}")
            retval.append(True)
            continue
        except KeyboardInterrupt:
            if interrupted:
                pass
            interrupted = True
            print(f"❌ Generation cancelled @ image №{i}.")
            print(f"✅ {i - 1} images generated.")
        except openai.BadRequestError as e: 
            print(f"❌ Generation failed, prompt triggered their filters.")
        except Exception as e:
            if len(str(e)) > 0:
                print(f"❌ Generation failed. Exception information: {e}")

        retval.append(False)

    if len(retval) == 0:
        retval.append(False)
    successes = sum(1 for r in retval if r)
    failures = sum(1 for r in retval if not r)
    formatted_time = datetime.now() - now
    hours, minutes, seconds = formatted_time.seconds // 3600, formatted_time.seconds // 60, formatted_time.seconds % 60
    requested_iterations = iterations
    if len(retval) != iterations:
        iterations = len(retval)
    print(f"⏱️ Took {hours} hours, {minutes} minutes and {seconds} seconds to generate {iterations} images,")
    print(f"✅ of which {successes} succeeded and {failures} failed.")
    if failures == 0:
        print(f"🎉 All images generated successfully!")
    else:
        print(f"🤔 Success rate: {successes / iterations * 100}%")
    if requested_iterations != iterations:
        print(f"📝 Note: {requested_iterations - iterations} images were not generated due to Unix signaling.")
    return all(retval)

def signal_handler(signum, _):
    global p
    global interrupted

    parent_shell = os.getppid()

    signame = signal.Signals(signum).name
    print(f"📻 Signal {signame} ({signum}) received,")
    if signum == signal.SIGINT or signum == signal.SIGTERM:
        print(f"❌ …will exit after image is generated.")
        interrupted = True
    else:
        if signum == signal.SIGUSR1 or signum == signal.SIGUSR2:
            print(f"🦊 ¿Qué?")
            try:
                os.system("ps")
            except:
                pass
            return
        msg = f"""
        📻 Signal {signum} received,
        ❌ \tand exiting to parent shell w/PID №{parent_shell}…\tgoodbye!
        """
        print(msg, file=stderr)
        exit(1)

def edit_image(image, brightness=None, contrast=None, sharpness=None):
    if brightness is not None:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast is not None:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if sharpness is not None:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image


_SEPIA_MATRIX = (
    # R
    0.393,
    0.769,
    0.189,
    # G
    0.349,
    0.686,
    0.168,
    # B
    0.272,
    0.534,
    0.131,
    # A
    1,
    1,
    1,
)

FILTERS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "grayscale": lambda i: i.convert("L"),
    "sepia": lambda i: i.convert("RGB")
    .convert("L")
    .convert(
        "RGB",
        matrix=_SEPIA_MATRIX,
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
    "outline": lambda i: i.filter(ImageFilter.CONTOUR).filter(ImageFilter.SMOOTH_MORE),
    "posterize": lambda i: i.convert("P", palette=Image.ADAPTIVE, colors=10),
    "solarize": lambda i: ImageOps.solarize(i, threshold=128),
    "invert": lambda i: ImageOps.invert(i),
    "flip": lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
    "rotate_90": lambda i: i.transpose(Image.ROTATE_90),
    "rotate_180": lambda i: i.transpose(Image.ROTATE_180),
    "rotate_270": lambda i: i.transpose(Image.ROTATE_270),
}


def apply_filter_choices(image, filter_name: str) -> Image.Image:
    filters = FILTERS
    if filter_name in filters:
        return filters[filter_name](image)
    return image


_HELP_ARGS = ("-h", "--help")
_HELP_KWARGS = {
    "help": "📖 Show this message and exit.",
    "is_flag": True,
}
_VERSION_ARGS = (None, "-v", "-V", "--version")
_VERSION_KWARGS = {
    "help": "📦 Show the version and exit.",
    "is_flag": True,
}


@click.group()
@click.version_option(*_VERSION_ARGS, **_VERSION_KWARGS)
@click.option(
    "--log-level",
    default="INFO",
    help="📝 The logging level to use",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.option(
    "--debug",
    is_flag=True,
    help="🐛 Enable debug mode",
)
@click.help_option(*_HELP_ARGS, **_HELP_KWARGS)
def cli(log_level, debug):
    """🖼️ Generate images from text using the OpenAI Dall.E API."""
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    logger.setLevel(log_level)


@cli.command("help")
@click.help_option(*_HELP_ARGS, **_HELP_KWARGS)
def help_():
    """📖 Show this message and exit."""
    for command in cli.commands.values():
        if command.name == "help":
            continue
        if not command.name:
            continue
        command_name = command.name.upper()
        print(click.style(f"{command_name}", fg="blue", bold=True))
        print(command.get_help(click.get_current_context()))
        print()


class SizeParamType(click.ParamType):
    name = "size"

    _valid_sizes = {
        "dall-e-2": ["256x256", "512x512", "1024x1024"],
        "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
    }
    t = typing

    def convert(
        self,
        value: str,
        param: Optional[click.Parameter] = None,
        ctx: Optional[click.Context] = None,
    ) -> Any:
        if not ctx or not hasattr(ctx, "params") or not "model" in ctx.params:
            model = "dall-e-3"
        else:
            model = ctx.params["model"]

        valid_sizes = self._valid_sizes[model]

        value = value.lower()

        if not "x" in value and all(map(lambda c: c.isdigit(), value)):
            value = f"{value}x{value}"

        if value in valid_sizes:
            return super().convert(value, param, ctx)
        else:
            self.fail(
                f"Invalid image size for model {model}."
                + f" Valid sizes are: {', '.join(valid_sizes)}.",
                param=param,
                ctx=ctx,
            )


class SizeParam(click.Option):
    _valid_sizes = SizeParamType._valid_sizes
    t = typing

    help = "📐 The size of the generated image."
    name = "size"
    type = SizeParamType()
    required = False
    nargs = 1
    multiple = False
    show_default = True
    opts = ("--size", "-s")
    prompt = False

    def __init__(self, *args, **kwargs):
        self.type = SizeParamType()
        super().__init__(
            *args,
            **kwargs,
            help=self.help,
            type=self.type,
            required=self.required,
            nargs=self.nargs,
            multiple=self.multiple,
            show_default=self.show_default,
            prompt=self.prompt,
        )

    def get_default(self, _: click.Parameter, **__) -> str | None:
        """Get the default value for the size parameter."""
        ctx = click.get_current_context(silent=True)
        if not ctx or not hasattr(ctx, "params"):
            return None
        elif not "model" in ctx.params:
            model = "dall-e-3"
        else:
            model = ctx.params["model"]

        return SizeParamType._valid_sizes[model][0]

    def make_metavar(self) -> str:
        return "SIZE"


@cli.command("generate")
@click.option(
    "--prompt",
    default="Suprise me",
    prompt=True,
    help="💬 The prompt to generate the image from.",
)
# valid image sizes for Dall-E 3 differ from Dall-E 2
@click.option(*SizeParam.opts, cls=SizeParam)
@click.option(
    "--save-path",
    type=click.Path(dir_okay=False, writable=True),
    help="💾 Save the generated image to the specified file path",
    default="images/output.png",
)
@click.option("--hide", is_flag=True, help="🖱️ Do not open the image after generation")
@click.option(
    "--model",
    default="dall-e-2",
    help="🤖 The model to use for generation",
    type=click.Choice(["dall-e-2", "dall-e-3"]),
)
@click.option(
    "--quality",
    default="standard",
    help="📷 The quality of the generated image",
    type=click.Choice(["standard", "hd"]),
)
@click.option(
    "--filter",
    default=None,
    multiple=True,
    show_choices=False,
    metavar="FILTER",
    help="🎨 The filter to apply to the image",
    type=click.Choice(
        [
            filter_name
            for filter_name in FILTERS.keys()
            if not filter_name.startswith("rotate")
        ]
    ),
)
@click.option(
    "--iterations",
    default=1,
    help="🔁 The number of images to generate",
    type=int,
)
@click.option(
    "--skip-exif",
    is_flag=True,
    help="📝 Do not append the prompt to the image as EXIF metadata",
)
@click.help_option(*_HELP_ARGS, **_HELP_KWARGS)
def generate(prompt, size, filter, iterations, hide, model, quality, skip_exif, save_path=None):
    """🖼️ Generate an image from a prompt using the Dall.E 2/3 API."""
    openai = configure_openai()
    generate_image(
        openai, prompt, size, filter, iterations, hide, model, quality, skip_exif, save_path
    )


@cli.command("edit")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--brightness", type=float, help="💡 The brightness of the image")
@click.option("--contrast", type=float, help="🌈 The contrast of the image")
@click.option("--sharpness", type=float, help="🔪 The sharpness of the image")
@click.help_option(*_HELP_ARGS, **_HELP_KWARGS)
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
def update_key():
    """🔐 Update the OpenAI API key."""
    config_file = path.expanduser("~/.openai/config.json")
    if not path.exists(config_file):
        makedirs(path.dirname(config_file), exist_ok=True)

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
            filter_name
            for filter_name in FILTERS.keys()
            if not filter_name.startswith("rotate")
        ]
    ),
    help="🎨 The filter to apply to the image",
    show_choices=True,
    show_default=True,
    required=True,
    multiple=True,
)
@click.option(
    "--save-path",
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default=".",
    help="The directory to save the filtered image. Defaults to the current directory.",
)
@click.help_option(*_HELP_ARGS, **_HELP_KWARGS)
def apply_filter(image_path, filters: list[str], save_path):
    """🦄 Apply filters and effects to an image."""
    image = Image.open(image_path)
    filtered_image_path = None
    if len(filters) == 0:
        print("❌ No filters specified.")
        return
    for filter_ in filters:
        filtered_image = apply_filter_choices(image, filter_)
        filtered_image_path = get_output_image_path(image_path, filter_, save_path)
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


if __name__ == "__main__":
    cli()

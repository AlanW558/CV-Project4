import os
import sys
import rich

_log_styles = {
    "BRPO": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="BRPO"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
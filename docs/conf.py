# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ABCMB"
copyright = "2026, Zilu Zhou, Cara Giovanetti, and Hongwan Liu"
author = "Zilu Zhou, Cara Giovanetti, and Hongwan Liu"

import sys

sys.path.append("..")

from abcmb.version import __version__

release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# # Make sure Sphinx can import your code
import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # if your package is in repo root

import re

from sphinx.util import logging

logger = logging.getLogger(__name__)


def format_method_summaries(app, what, name, obj, options, lines):
    """
    Rewrite blocks like

        Recombination Unrelated Methods:
        rho_tot : Compute total energy density ...
        P_tot : Compute total pressure ...

    or

        Background Quantities:
        -------
        rho_tot : Compute total energy density ...
        P_tot : Compute total pressure ...

    into valid reST bullet lists, so Sphinx preserves line breaks.

    This function detects any section that uses hyphen underlining (NumPy convention)
    or ends with "Methods:" and applies consistent formatting.

    It avoids formatting prose sections (like "Notes:") where patterns like
    "IDEA: text" or "TODO: text" should remain as plain text.
    """
    out = []
    in_custom_block = False
    in_methods_block = False
    in_prose_section = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if next line is a hyphen underline (NumPy convention)
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Detect hyphen underlining: a line of hyphens matching the header length
            if re.match(r"^\s*-+\s*$", next_line) and line.strip():
                # This is a section header with hyphen underlining
                in_custom_block = True

                # Check if this is a Methods section
                in_methods_block = bool(re.search(r"Methods:", line, re.IGNORECASE))

                # Check if this is a prose section (Notes, Examples, etc.)
                in_prose_section = bool(
                    re.search(
                        r"(Notes|Examples|See Also|References|Warnings):",
                        line,
                        re.IGNORECASE,
                    )
                )

                # Output the header line with bold formatting
                out.append(f"**{line.strip()}**")
                # Skip the hyphen line (don't output it)
                i += 1
                # Add blank line for proper reST parsing
                out.append("")
                i += 1
                continue

        # Also detect headings that end with "Methods:" (original behavior)
        if re.match(r"^\s*.*Methods:\s*$", line):
            in_custom_block = True
            in_methods_block = True
            in_prose_section = False

            # Output the header line with bold formatting
            out.append(f"**{line.strip()}**")

            # IMPORTANT: add a blank line so the next lines
            # can become a proper bullet list in reST
            out.append("")
            i += 1
            continue

        # Blank line: close the block
        if line.strip() == "":
            in_custom_block = False
            in_methods_block = False
            in_prose_section = False
            out.append(line)
            i += 1
            continue

        if in_custom_block and not in_prose_section:
            # Look for "name : description ..."
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
            if m:
                ident, desc = m.groups()
                # Skip only specific prose markers like "IDEA:", "TODO:", "NOTE:"
                if ident.upper() in ("IDEA", "TODO", "NOTE", "WARNING", "FIXME"):
                    # This looks like prose text, not a method/attribute name
                    out.append(line)
                    i += 1
                    continue

                # For Methods sections, use bullet with field-like format
                # For other sections, keep inline format
                if in_methods_block:
                    out.append(f"* ``{ident}`` :")
                    out.append(f"      {desc}")
                else:
                    out.append(f"* ``{ident}``  {desc}")
                i += 1
                continue

        # default passthrough
        out.append(line)
        i += 1

    # mutate the list in-place so autodoc uses our version
    lines[:] = out

    # optional debug: shows up during sphinx-build
    logger.debug(f"[format_method_summaries] processed {name} ({what})")


def skip_equinox_field_attributes(app, what, name, obj, skip, options):
    """
    Skip class attributes that are equinox fields to prevent duplicate documentation.
    These are already documented in the class docstring's Fields section.
    """

    # Only process attributes
    if what not in ("attribute", "data"):
        return skip

    # Try to get the class that owns this attribute
    try:
        # Get the parent class name from the full qualified name
        parts = name.split(".")
        if len(parts) < 2:
            return skip

        # Import the module and get the class
        module_name = ".".join(parts[:-2])
        class_name = parts[-2]
        attr_name = parts[-1]

        import importlib

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name, None)

        if cls is None:
            return skip

        # Check if this attribute is defined at the class level (not instance level)
        # Class-level attributes defined with eqx.field() will be in __annotations__
        if hasattr(cls, "__annotations__") and attr_name in cls.__annotations__:
            # This is a class attribute, skip it
            return True

    except Exception:
        # If anything goes wrong, don't skip
        pass

    return skip


def truncate_array_defaults(
    app, what, name, obj, options, signature, return_annotation
):
    """
    Truncate large array default values in function/method signatures.

    This handler processes signatures to replace long array defaults with a
    truncated format: [first_value, ..., last_value]

    Arrays with more than 3 elements are truncated to show only the first
    and last elements.
    """
    if signature is None:
        return None

    import re

    def truncate_array_in_signature(sig_str):
        """
        Truncate array representations in signature strings.
        Handles Array([...], dtype=...) format from JAX/NumPy.
        """

        # Pattern to match Array([...], dtype=...)
        # This needs to handle arrays that may span multiple conceptual elements
        def replace_array(match):
            full_match = match.group(0)

            # Extract the array content between [ and ]
            array_start = full_match.find("[")
            array_end = full_match.rfind("]")

            if array_start == -1 or array_end == -1:
                return full_match

            array_content = full_match[array_start + 1 : array_end]

            # Count commas at depth 0 to determine number of elements
            depth = 0
            comma_count = 0
            for char in array_content:
                if char in "([":
                    depth += 1
                elif char in ")]":
                    depth -= 1
                elif char == "," and depth == 0:
                    comma_count += 1

            num_elements = comma_count + 1

            # Only truncate if more than 3 elements
            if num_elements > 3:
                # Find first element (up to first comma at depth 0)
                depth = 0
                first_comma_idx = -1
                for i, char in enumerate(array_content):
                    if char in "([":
                        depth += 1
                    elif char in ")]":
                        depth -= 1
                    elif char == "," and depth == 0:
                        first_comma_idx = i
                        break

                # Find last comma at depth 0 (working backwards)
                depth = 0
                last_comma_idx = -1
                for i in range(len(array_content) - 1, -1, -1):
                    char = array_content[i]
                    if char in ")]":
                        depth += 1
                    elif char in "([":
                        depth -= 1
                    elif char == "," and depth == 0:
                        last_comma_idx = i
                        break

                if first_comma_idx > 0 and last_comma_idx > first_comma_idx:
                    first_val = array_content[:first_comma_idx].strip()
                    last_val = array_content[last_comma_idx + 1 :].strip()

                    # Check if there's a dtype specification after the array
                    dtype_match = re.search(r",\s*dtype=\w+", full_match[array_end:])
                    dtype_str = dtype_match.group(0) if dtype_match else ""

                    # Reconstruct with truncation
                    return f"Array([{first_val}, ..., {last_val}]{dtype_str})"

            return full_match

        # Match Array([...], dtype=...) format - be greedy to capture full array
        # Use a more flexible pattern that handles multi-line arrays
        sig_str = re.sub(
            r"Array\(\[(?:[^\[\]]|\[[^\]]*\])*\](?:,\s*dtype=\w+)?\)",
            replace_array,
            sig_str,
            flags=re.DOTALL,
        )

        return sig_str

    signature = truncate_array_in_signature(signature)
    return signature, return_annotation


def setup(app):
    # connect our transformer so it runs on every autodoc'd object
    app.connect("autodoc-process-docstring", format_method_summaries)
    # Skip equinox field attributes
    app.connect("autodoc-skip-member", skip_equinox_field_attributes)
    # Truncate large array defaults in signatures
    app.connect("autodoc-process-signature", truncate_array_defaults)


templates_path = ["_templates"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
intersphinx_mapping = {
    "hyrex": ("https://hyrex.readthedocs.io/en/latest/", None),
}
exclude_patterns = [
    "ABCMB/hyrex",  # exclude the subpackage
    "../ABCMB/hyrex",  # relative path safety
    "HyRex",  # in case of capitalization variants
    "ABCMB/linx",
    "../ABCMB/linx",
    "LINX",
    "_build",
    "Thumbs.db",
    ".DS_Store",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

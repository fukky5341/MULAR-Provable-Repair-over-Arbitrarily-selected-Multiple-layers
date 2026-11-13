"""Run the full repair benchmarks with fault localization enabled.

This entry point intentionally uses the same experiment pipeline and command
line options as ``test.py``. Its only different defaults are:

* fault localization is enabled;
* ``original_arachne`` is the selected localization mode.
* three Pareto fronts are selected.
* region-derived negative samples are used.
* Original-Arachne-selected parameters are used rather than the random control.

Both defaults can still be overridden explicitly on the command line.

running:
uv run python test_fl.py --exe-id 1 --arachne-layer-order selected_parameter_count --repair-layer-config m9_12_16

running with random selection:
uv run python test_fl.py \
  --exe-id 1 \
  --repair-layer-config m9_12_16 \
  --arachne-parameter-selection random_matched \
  --arachne-random-seed 0

running with random selection matched separately within each layer:
uv run python test_fl.py \
  --exe-id 1 \
  --repair-layer-config m9_12_16 \
  --arachne-parameter-selection random_layer_matched \
  --arachne-random-seed 0
"""

from pathlib import Path
import runpy
import sys


def _has_option(arguments, option):
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


# repairable layers and max subrepair loops
repairable_layers_and_max_loops = {
    "m9_2_16": ([2, 16], 3),
    "m9_10_14": ([10, 14], 4),
    "m9_10_16": ([10, 16], 3),
    "m9_12_16": ([12, 16], 4),
    "m9_14_16": ([14, 16], 5),
}


def main():
    forwarded_arguments = list(sys.argv[1:])
    defaults = []
    if not _has_option(forwarded_arguments, "--perform_FL"):
        defaults.extend(("--perform_FL", "True"))
    if not _has_option(forwarded_arguments, "--fl-mode"):
        defaults.extend(("--fl-mode", "original_arachne"))
    if not _has_option(forwarded_arguments, "--pareto_round"):
        defaults.extend(("--pareto_round", "10"))
    if not _has_option(forwarded_arguments, "--arachne-negative-source"):
        defaults.extend(("--arachne-negative-source", "test"))  # region, test
    if not _has_option(forwarded_arguments, "--arachne-parameter-selection"):
        defaults.extend(("--arachne-parameter-selection", "original"))

    sys.argv = [sys.argv[0], *defaults, *forwarded_arguments]
    runpy.run_path(
        str(Path(__file__).with_name("test.py")),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()

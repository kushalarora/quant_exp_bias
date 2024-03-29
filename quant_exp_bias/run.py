#!/usr/bin/env python
import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position
from quant_exp_bias.commands.sample_oracle import SampleOracle
from quant_exp_bias.commands.quantify_exposure_bias import QuantifyExposureBias
from quant_exp_bias.commands.compute_nll import ComputeNLLScore
from allennlp.common.util import import_module_and_submodules

def run():
    import_module_and_submodules("lmpl")
    import_module_and_submodules("quant_exp_bias")
    main(prog="allennlp")

if __name__ == "__main__":
    run()

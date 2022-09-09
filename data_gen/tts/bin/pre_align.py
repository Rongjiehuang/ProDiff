import os

os.environ["OMP_NUM_THREADS"] = "1"

import importlib
from utils.hparams import set_hparams, hparams


def pre_align():
    assert hparams['pre_align_cls'] != ''

    pkg = ".".join(hparams["pre_align_cls"].split(".")[:-1])
    cls_name = hparams["pre_align_cls"].split(".")[-1]
    process_cls = getattr(importlib.import_module(pkg), cls_name)
    process_cls().process()


if __name__ == '__main__':
    set_hparams()
    pre_align()

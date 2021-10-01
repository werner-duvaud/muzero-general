from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from dm_control.utils import io as resources

# make sawyer meshes easy to import as a batch
_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_SUITE_DIR = osp.join(_SUITE_DIR, 'meshes')
_FILENAMES = [osp.join(_SUITE_DIR, f) for f in os.listdir(_SUITE_DIR) if f.endswith('.stl') or f.endswith('.png')]

ASSETS = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename))
          for filename in _FILENAMES}

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for the libtpu library."""
from importlib import metadata
import os

__version__ = metadata.version(__name__)


def get_library_path() -> str:
  return os.path.join(os.path.dirname(__file__), 'libtpu.so')


def configure_library_path():
  if not os.environ.get('TPU_LIBRARY_PATH'):
    os.environ['TPU_LIBRARY_PATH'] = get_library_path()


# on import of libtpu set the library path if not already configured.
configure_library_path()

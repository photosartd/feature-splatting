from typing import Optional

import viser


class GlobalRegistry:
    viser_server: Optional[viser.ViserServer] = None

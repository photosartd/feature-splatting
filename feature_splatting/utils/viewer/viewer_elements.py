from nerfstudio.viewer.viewer_elements import ViewerElement
from viser import ViserServer, GuiButtonHandle

from feature_splatting.utils.viewer.registry import GlobalRegistry

class ViserServerBackdoor(ViewerElement[bool]):
    """A backdoor to send the viser_server to some part of the app

    Args:
        name: The name of the button
        cb_hook: The function to call when the button is pressed
        disabled: If the button is disabled
        visible: If the button is visible
    """
    NAME: str = "viser_server_backdoor"
    gui_handle: GuiButtonHandle

    def __init__(self):
        super().__init__(self.NAME, disabled=True, visible=False)

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        self.gui_handle = viser_server.gui.add_button(label=self.name, disabled=self.disabled, visible=self.visible)

    def install(self, viser_server: ViserServer) -> None:
        self._create_gui_handle(viser_server)

        assert self.gui_handle is not None
        self.gui_handle.on_click(lambda _: self.cb_hook(self))

        # Backdoor
        GlobalRegistry.viser_server = viser_server

##
#
# Use Mujoco's Viewer to show the model.
#
# Note: need mujoco python
##

import mujoco
from mujoco.viewer import launch

# Load the model from XML
xml_file = "./models/hotdog_man.xml"

# load and launch the model
model =  mujoco.MjModel.from_xml_path(xml_file)
launch(model)
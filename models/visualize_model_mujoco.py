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
# xml_file = "./models/cart_pole.xml"
# xml_file = "./models/acrobot.xml"

# load and launch the model
model =  mujoco.MjModel.from_xml_path(xml_file)
launch(model)
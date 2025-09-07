##
#
# Indices for the model
#
##

####################################################################
# PLANAR HOPPER
####################################################################

class Hopper_IDX:

    # full state indices
    class STATE:
        
        POS_X = 0
        POS_Z = 1
        EUL_Y = 2
        POS_LEG = 3

        VEL_X = 4
        VEL_Z = 5
        ANG_Y = 6
        VEL_LEG = 7

        SIZE = 8

    # generalized positions
    class POS:

        POS_X = 0
        POS_Z = 1
        EUL_Y = 2
        POS_LEG = 3

        SIZE = 4

    # generalized velocities
    class VEL:

        VEL_X = 0
        VEL_Z = 1
        ANG_Y = 2
        VEL_LEG = 3

        SIZE = 4

    # actuated joints
    class JOINT:

        LEG = 0

        SIZE = 1

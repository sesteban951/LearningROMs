##
#
# Indices for XML models
#
##

####################################################################
# HOTDOGMAN PLANAR BIPED
####################################################################

class HotdogMan_IDX:

    # full state indices
    class STATE:
        
        POS_X = 0
        POS_Z = 1
        EUL_Y = 2
        POS_LH = 3
        POS_LK = 4
        POS_RH = 5
        POS_RK = 6

        VEL_X = 7
        VEL_Z = 8
        ANG_Y = 9
        VEL_LH = 10
        VEL_LK = 11
        VEL_RH = 12
        VEL_RK = 13
        
        SIZE = 14

    # generalized positions
    class POS:

        POS_X = 0
        POS_Z = 1
        EUL_Y = 2
        POS_LH = 3
        POS_LK = 4
        POS_RH = 5
        POS_RK = 6

        SIZE = 7

    # generalized velocities
    class VEL:

        VEL_X = 0
        VEL_Z = 1
        ANG_Y = 2
        VEL_LH = 3
        VEL_LK = 4
        VEL_RH = 5
        VEL_RK = 6

        SIZE = 7

    # actuated joints
    class JOINT:

        POS_LH = 0
        POS_LK = 1
        POS_RH = 2
        POS_RK = 3

        SIZE = 4
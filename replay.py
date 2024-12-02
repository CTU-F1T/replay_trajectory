#!/usr/bin/env python
# replay.py
"""Replay trajectory data to simulate a driving vehicle.

While this should be a pretty simple script, it might do some slightly more
complex tasks in the future.
"""
######################
# Imports & Globals
######################

import argparse
import csv
import numpy
import sys

from autopsy.node import Node
from autopsy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

try:
    # Requires autopsy>=0.11
    from autopsy.helpers import Execute
except ImportError:
    import autopsy

    sys.stderr.write(
        "This script requires autopsy>=0.11, but '%s' is installed.\n"
        % autopsy.__version__
    )

    raise


from nav_msgs.msg import Odometry


# Global variables
PARSER = argparse.ArgumentParser(
    prog = "replay.py",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description = """
Script for replaying trajectory data to simulate a driving vehicle.
    """,
)


# Arguments
PARSER.add_argument(
    "input_file",

    help = "Path to the trajectory (.csv) file.",
    type = argparse.FileType("r"),
)

PARSER.add_argument(
    "-r",

    dest = "rate",
    metavar = "RATE",
    help = "Publishing rate [Hz] (default %(default)d).",
    type = int,
    default = 10,
)

PARSER.add_argument(
    "--frame_id",

    help = "Frame id of the simulated car (default %(default)s).",
    type = str,
    default = "map",
)



PARSER.add_argument(
    "-v",

    help = "Increase verbosity.",
    action = "store_true",
)


######################
# Trajectory object
######################

class Trajectory(object):
    """Trajectory object."""

    def __init__(
        self,
        x = None,
        y = None,
        t = None,
        v = None,
        lap_time = None,
        **kwargs
    ):
        """Initialize the object."""
        self._x = numpy.asarray(x)
        self._y = numpy.asarray(y)
        self._t = numpy.asarray(t)
        self._t2 = numpy.asarray(t + [lap_time])
        self._v = numpy.asarray(v)

        self._lap_time = lap_time


    @staticmethod
    def from_dict(d, **mappings):
        """Create Trajectory object from a dictionary and given mappings.

        Arguments:
        d -- dictionary with lists of data
        **mappings -- keyword dictionary with string creating mappings for
                      the given dictionary

        Note: This is separated so some additional check may be performed.
        """
        return Trajectory(
            **{
                key: d[value] for key, value in mappings.items()
            }
        )


    @property
    def size(self):
        """Obtain the size of the trajectory.

        Returns:
        size -- number of points in the Trajectory, int
        """
        return len(self._x)


    def closest_point_time(self, time):
        """Find a closest point on the trajectory to the specified time.

        Arguments:
        time -- time to be closest to

        Returns:
        time_distance -- time delta between the points
        i -- index of the closest point

        TODO: Solve the endpoints as they are not properly used.
        """
        if self._t[0] is None:
            raise ValueError(
                "Unable to find time-closest point as trajectory "
                "lacks timestamps."
            )

        _distances = numpy.sqrt(
            numpy.power(self._t2 - (time % self._lap_time), 2)
        )
        _min_i = numpy.argmin(_distances) % self.size
        return _distances[_min_i], _min_i


######################
# Node class
######################

class ReplayNode(Node):
    """ROS Node for replaying the trajectory."""

    def __init__(self, trajectory, rate, frame_id = "map", **kwargs):
        """Initialize the node."""
        super(ReplayNode, self).__init__("replay_trajectory", **kwargs)

        self._trajectory = trajectory
        self._frame_id = frame_id

        self._pub = self.create_publisher(
            Odometry, "/odom", qos_profile = QoSProfile(
                depth = 1, durability = DurabilityPolicy.VOLATILE,
                reliability = ReliabilityPolicy.BEST_EFFORT
            )
        )

        self._loop = self.create_timer(1. / rate, self.loop).run()


    def loop(self, *args, **kwargs):
        """Timer loop for publishing the data."""
        cur_time = self.Time.now()

        _, index = self._trajectory.closest_point_time(
            cur_time.nanoseconds / 1.e9
        )

        msg = Odometry()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = cur_time.to_msg()
        msg.pose.pose.position.x = self._trajectory._x[index]
        msg.pose.pose.position.y = self._trajectory._y[index]

        self._pub.publish(msg)


######################
# Utilities
######################

def read_csv(fh):
    """Read contents of a csv file.

    Arguments:
    fh -- file handle to opened csv file

    Returns:
    dict of lists, keys are csv columns
    """
    dr = csv.DictReader(fh)

    data = {field: [] for field in dr.fieldnames}

    for line in dr:
        for key, value in line.items():
            try:
                value = float(value)
            except ValueError:
                pass

            data[key].append(value)

    return data


######################
# Main
######################

if __name__ == "__main__":
    args = PARSER.parse_args()

    # 1. Read data
    data = read_csv(args.input_file)
    lap_time = data["t_s"][0]
    data["t_s"][0] = .0

    # 2. Create trajectory object
    t = Trajectory(
        x = data["x_m"], y = data["y_m"], v = data["v_mps"], t = data["t_s"],
        lap_time = lap_time
    )

    # 3. Run the node
    Execute(
        ReplayNode,
        rate = args.rate, trajectory = t, frame_id = args.frame_id
    )

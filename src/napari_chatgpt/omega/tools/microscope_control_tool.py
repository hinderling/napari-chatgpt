"""A tool for planing and executing an Acquitision on the Microscope Hardware."""
from pathlib import Path

from napari import Viewer

from napari_chatgpt.omega.tools.napari_base_tool import NapariBaseTool
from napari_chatgpt.utils.dynamic_import import dynamic_import

from pymmcore_plus import CMMCorePlus
from useq import MDAEvent, MDASequence
import zarr
import numpy as np

_microscope_control_prompt = """
Task:
Given a plain text request, you competently write a python function create_acquisition(pm,viewer) that calls the MDASequence() function.
The function create_acquisition() takes no inputs and returns the output of MDASequence.
The dictionary of acquisition events is used to create an acquisition plan.
The acquisition plan is executed by the microscope hardware object pm.

Only fill provide the arguments that are requested by the user. E.g. if no channels are provided, leave out the channels argument. 
If not multiple images are requested, leave out the timepoints argument.
Never invent arguments that are not requested by the user.

{generic_codegen_instructions}

Below is a description of the function MDASequences() that you need to call.

A sequence of MDA (Multi-Dimensional Acquisition) events.

    This is the core object in the `useq` library, and is used define a sequence of
    events to be run on a microscope. It object may be constructed manually, or from
    file (e.g. json or yaml).

    The object itself acts as an iterator for [`useq.MDAEvent`][] objects:

    Attributes
    ----------
    metadata : dict
        A dictionary of user metadata to be stored with the sequence.
    axis_order : str
        The order of the axes in the sequence. Must be a permutation of `"tpgcz"`. The
        default is `"tpgcz"`.
    stage_positions : tuple[Position, ...]
        The stage positions to visit. (each with `x`, `y`, `z`, `name`, and `sequence`,
        all of which are optional).
    grid_plan : GridFromEdges, GridRelative, NoGrid
        The grid plan to follow. One of `GridFromEdges`, `GridRelative` or `NoGrid`.
    channels : tuple[Channel, ...]
        The channels to acquire. see `Channel`.
    time_plan : MultiPhaseTimePlan | TIntervalDuration | TIntervalLoops        | TDurationLoops | NoT
        The time plan to follow. One of `TIntervalDuration`, `TIntervalLoops`,
        `TDurationLoops`, `MultiPhaseTimePlan`, or `NoT`
    z_plan : ZTopBottom | ZRangeAround | ZAboveBelow | ZRelativePositions |        ZAbsolutePositions | NoZ
        The z plan to follow. One of `ZTopBottom`, `ZRangeAround`, `ZAboveBelow`,
        `ZRelativePositions`, `ZAbsolutePositions`, or `NoZ`.


To use function MDASequence() simply import it:
from useq import MDASequence

Examples (don't include in your answer!!!):
--------
UserRequest: Acquire two loops with 0.1 second interval, capture a 2x2 grid and 3 z steps, in the DAPI channel with 1ms exposure.
AgentResponse:

def create_acquisition():
    from useq import MDASequence, Position, Channel, TIntervalDuration
    seq = MDASequence(
        time_plan={{\"interval\": 0.1, \"loops\": 2}},
        stage_positions=[(1, 1, 1)],
        grid_plan={{\"rows\": 2, \"cols\": 2}},
        z_plan={{\"range\": 3, \"step\": 1}},
        channels=[{{\"config\": \"DAPI\", \"exposure": 1}}]
    )
    return seq

UserRequest: Acquire an image.
AgentResponse:
def create_acquisition():
    from useq import MDASequence, Position, Channel, TIntervalDuration
    seq = MDASequence(
        time_plan={{\"interval\": 0, \"loops\": 1}},
    )
    return seq


__________________
USEQ DOCUMENTATION:
__________________

class TIntervalLoops(TimePlan):
    \"""Define temporal sequence using interval and number of loops.

    Attributes
    ----------
    interval : str | timedelta
        Time between frames.
    loops : PositiveInt
        Number of frames.
    prioritize_duration : bool
        If `True`, instructs engine to prioritize duration over number of frames in case
        of conflict. By default, `False`.
    \"""


class TDurationLoops(TimePlan):
    \"""Define temporal sequence using duration and number of loops.

    Attributes
    ----------
    duration : str | timedelta
        Total duration of sequence.
    loops : PositiveInt
        Number of frames.
    prioritize_duration : bool
        If `True`, instructs engine to prioritize duration over number of frames in case
        of conflict. By default, `False`.
    \"""

class TIntervalDuration(TimePlan):
    \"""Define temporal sequence using interval and duration.

    Attributes
    ----------
    interval : str | timedelta
        Time between frames.
    duration : str | timedelta
        Total duration of sequence.
    prioritize_duration : bool
        If `True`, instructs engine to prioritize duration over number of frames in case
        of conflict. By default, `True`.
    \"""


class NoT(TimePlan):
    \"""Don't acquire a time sequence.\"""

    def deltas(self) -> Iterator[datetime.timedelta]:
        yield from ()

class Channel(FrozenModel):
    \"""Define an acquisition channel.

    Attributes
    ----------
    config : str
        Name of the configuration to use for this channel, (e.g. `"488nm"`, `"DAPI"`,
        `"FITC"`).
    group : str
        Optional name of the group to which this channel belongs. By default,
        `"Channel"`.
    exposure : PositiveFloat | None
        Exposure time in seconds. If not provided, implies use current exposure time.
        By default, `None`.
    do_stack : bool
        If `True`, instructs engine to include this channel in any Z stacks being
        acquired. By default, `True`.
    z_offset : float
        Relative Z offset from current position, in microns. By default, `0`.
    acquire_every : PositiveInt
        Acquire every Nth frame (if acquiring a time series). By default, `1`.
    camera: str | None
        Name of the camera to use for this channel. If not provided, implies use
        current camera. By default, `None`.
    \"""

    class Position(FrozenModel):
    \"""Define a position in 3D space.

    Any of the attributes can be `None` to indicate that the position is not
    defined. This is useful for defining a position relative to the current
    position.

    Attributes
    ----------
    x : float | None
        X position in microns.
    y : float | None
        Y position in microns.
    z : float | None
        Z position in microns.
    name : str | None
        Optional name for the position.
    sequence : MDASequence | None
        Optional MDASequence relative this position.
    \"""

    class GridFromEdges(_GridPlan):
    \"""Yield absolute stage positions to cover a bounded area...

    ...defined by setting the stage coordinates of the top, left,
    bottom and right edges.

    Attributes
    ----------
    top : float
        top stage position of the bounding area
    left : float
        left stage position of the bounding area
    bottom : float
        bottom stage position of the bounding area
    right : float
        right stage position of the bounding area
    \"""

    top: float
    left: float
    bottom: float
    right: float

    def _nrows(self, dy: float) -> int:
        total_height = abs(self.top - self.bottom) + dy
        return math.ceil(total_height / dy)

    def _ncolumns(self, dx: float) -> int:
        total_width = abs(self.right - self.left) + dx
        return math.ceil(total_width / dx)

    def _offset_x(self, dx: float) -> float:
        return min(self.left, self.right)

    def _offset_y(self, dy: float) -> float:
        return max(self.top, self.bottom)


class GridRelative(_GridPlan):
    \"""Yield relative delta increments to build a grid acquisition.

    Attributes
    ----------
    rows: int
        Number of rows.
    columns: int
        Number of columns.
    relative_to : RelativeTo
        Point in the grid to which the coordinates are relative. If "center", the grid
        is centered around the origin. If "top_left", the grid is positioned such that
        the top left corner is at the origin.
    \"""


class NoGrid(_GridPlan):
    \"""Don't acquire a grid.\"""

__________________
END USEQ DOCUMENTATION
__________________

    

IMPORTANT INSTRUCTIONS: 
- DO NOT include code for the function 'MDASequence()' in your answer.
- INSTEAD, DIRECTLY call the function 'MDASequence()' provided above after import.
- Assume that the function 'MDASequence()' is available and within scope of your code.
- Response is only the python function: create_acquisition()->seq.
- DO NOT ASSUME any channel definitions that are not provided, if no channels are provided leave the argument empty.
- E.g. if not explicitly brightfield or BF is requested, do not include it in the channels argument.
- In any case do not invent new arguments, just fill in the minimal number required to satisfy the request.



TASK: With help of the above documentation, write a function that satisfies this USER REQUEST:
    

{input}


Answer in markdown with a single function create_acquisition()->seq that creates the sequence of planned acquisition events.
"""


class MicroscopeControlTool(NapariBaseTool):
    """A tool for acquiring images, they can also be multi dimensional (multiple timepoints for timelapses, z-stacks, multiple positions or FOVs, multiple channels)."""

    name = "MicroscopeControlTool"
    description = (
        "Forward plain text requests to this tool when you need to acquire images (with a microscope)."
        "This tool has the highest priority when the request pertains to acuiring images, with single or multiple timepoints/positions/channels."
       # "This tool operates on image layers present in the already instantiated napari viewer."
    )
    prompt = _microscope_control_prompt

    # generic_codegen_instructions: str = ''

    def _run_code(self, query: str, code: str, viewer: Viewer) -> str:
        # prepare code:
        print(code)
        code = super()._prepare_code(code)
       # print(code)
        # cellpose_code = _get_seg_code('cellpose')
        #
        # # combine generated code and functions:
        # code = cellpose_code +'\n\n' + code

        # Load the code as module:
        loaded_module = dynamic_import(code)

        # get the function:
        create_acquisition = getattr(loaded_module, 'create_acquisition')


        # Run generated code:
        sequence = create_acquisition()


        # Add to viewer:
###        viewer.add_labels(segmented_image, name='segmented')
        mmc = CMMCorePlus.instance()  
        mmc.loadSystemConfiguration()  #  load demo configuration 

        camera_width = mmc.getImageWidth()
        camera_height = mmc.getImageHeight()

        # Create an empty zarr array with the desired shape and chunk size
        shape = sequence.shape+(camera_width,camera_height)
        chunks = tuple(np.repeat(1,len(sequence.shape)))+(512,512)
        dtype = 'float32'

        store = zarr.DirectoryStore('/tmp/tmp.zarr')
        #store = zarr.DirectoryStore('/tmp/tmp.zarr')
        zarr_array = zarr.zeros(shape, dtype=dtype, chunks=chunks, store=store, overwrite=True)


        # connect callback using a decorator 
        #@mmc.mda.events.frameReady.connect
        def new_frame(img: np.ndarray, event: MDAEvent):
            index = list(event.index.values()) #get index of current frame (e.g. (2,5)(t,c))
            zarr_array[tuple(index) + (slice(None),) * 2] = img #add image to data array at correct index

        # Connect the callback to the frameReady event
        mmc.mda.events.frameReady.connect(new_frame)

        viewer.add_image(zarr_array)
        # run the sequence in a separate thread 
        mmc.run_mda(sequence)
        
        return f"Success: acquisition completed! Successfully added image to viewer. Shape of image is {zarr_array.shape}."
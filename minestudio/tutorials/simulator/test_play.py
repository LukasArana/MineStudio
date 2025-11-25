import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    PlayCallback, RecordCallback, PointCallback, PlaySegmentCallback, PrevActionCallback
)
from minestudio.simulator.utils.gui import RecordDrawCall, CommandModeDrawCall, SegmentDrawCall
from functools import partial
if __name__ == '__main__':
    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[
            PlayCallback(agent_generator=None, extra_draw_call=[RecordDrawCall, CommandModeDrawCall]),
            PrevActionCallback(), 
            RecordCallback(record_path='./output', recording=False),
        ]
    )
    obs, info = sim.reset()
    terminated = False

    #action = sim.action_space.sample()
    while not terminated:
        #print(action)
        obs, reward, terminated, truncated, info = sim.step(action = None)

    sim.close()

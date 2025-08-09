from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

from builtins import range
from past.utils import old_div
import MalmoPython
import random
import time
import logging
import struct
import socket
import os
import sys
import malmoutils

# NEU: Module für die Visualisierung importieren
import numpy as np
import matplotlib.pyplot as plt


malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)
recordingsDirectory = malmoutils.get_recordings_directory(agent_host)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------------------------------------------------------------------------------

def processFrame( frame ):
    '''Diese Funktion analysiert die rohen Pixel, um eine Steuerungsentscheidung zu treffen.'''
    global current_yaw_delta_from_depth
    y = int(old_div(video_height, 2))
    rowstart = y * video_width
    v = 0
    v_max = 0
    v_max_pos = 0
    for x in range(0, video_width):
        nv = frame[(rowstart + x) * 4 + 3] # Extrahiert Tiefe aus dem 4. Kanal
        if nv > v_max or x == 0:
            v_max = nv
            v_max_pos = x
        v = nv
    
    if v_max < 255:
        current_yaw_delta_from_depth = (old_div(float(v_max_pos), video_width)) - 0.5
    else:
        if current_yaw_delta_from_depth < 0:
            current_yaw_delta_from_depth = -1
        else:
            current_yaw_delta_from_depth = 1

#----------------------------------------------------------------------------------------------------------------------------------

current_yaw_delta_from_depth = 0
video_width = 432
video_height = 240
   
missionXML = '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    
      <About>
        <Summary>Run the maze!</Summary>
      </About>
      
     <ServerSection>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
            <MazeDecorator>
                <SizeAndPosition length="20" width="20" xOrigin="0" yOrigin="215" zOrigin="410" height="180"/>
                <GapProbability variance="0.1">0.9</GapProbability>
                <Seed>random</Seed>
                <MaterialSeed>random</MaterialSeed>
                <AllowDiagonalMovement>false</AllowDiagonalMovement>
                <StartBlock fixedToEdge="true" type="emerald_block" height="1"/>
                <EndBlock fixedToEdge="true" type="redstone_block" height="12"/>
                <PathBlock type="glowstone" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="1"/>
                <FloorBlock type="air"/>
                <SubgoalBlock type="beacon"/>
                <GapBlock type="stained_hardened_clay" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="3"/>
            </MazeDecorator>
            <ServerQuitFromTimeUp timeLimitMs="30000"/>
            <ServerQuitWhenAnyAgentFinishes />
        </ServerHandlers>
    </ServerSection>

    <AgentSection>
        <Name>Jason Bourne</Name>
        <AgentStart>
            <Placement x="-203.5" y="81.0" z="217.5"/>
        </AgentStart>
        <AgentHandlers>
            <VideoProducer want_depth="true">
                <Width>''' + str(video_width) + '''</Width>
                <Height>''' + str(video_height) + '''</Height>
            </VideoProducer>
            <ContinuousMovementCommands turnSpeedDegs="720" />
            <AgentQuitFromTouchingBlockType>
                <Block type="redstone_block"/>
            </AgentQuitFromTouchingBlockType>
        </AgentHandlers>
    </AgentSection>
  </Mission>'''

validate = True
my_mission = MalmoPython.MissionSpec( missionXML, validate )

agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 30000

# NEU: Plot-Fenster *einmal* vor allen Missions-Wiederholungen initialisieren
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Tiefenbild (Depth Map)")
img_plot = None

my_mission_record = MalmoPython.MissionRecordSpec()
if recordingsDirectory:
    my_mission_record.setDestination(recordingsDirectory + "//" + "Mission_1.tgz")

for iRepeat in range(num_reps):
    logger.info('Mission %s', iRepeat + 1)
    
    # Optional: Video für diese Wiederholung aufzeichnen
    if recordingsDirectory:
        my_mission_record.setDestination(recordingsDirectory + "//" + "Mission_" + str(iRepeat + 1) + ".tgz")

    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                logger.error("Error starting mission: %s" % e)
                exit(1)
            else:
                time.sleep(2)

    logger.info("Waiting for the mission to start")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
    print()

    agent_host.sendCommand( "move 1" )

    # Hauptschleife:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_video_frames_since_last_state > 0:
            logger.info("Got frame!")
            frame = world_state.video_frames[-1]

            # NEU: Korrigierte Logik zur Anzeige des Tiefenbildes
            # Pixel-Liste in ein 4-Kanal-Bild (BGRA) umformen
            pixels = np.array(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 4)
            # Den 4. Kanal (Index 3) extrahieren, der die Tiefen-Info enthält
            depth_image = pixels[:, :, 3]
            
            # Bild im Fenster anzeigen
            if img_plot is None:
                img_plot = ax.imshow(depth_image, cmap='gray_r')
            else:
                img_plot.set_data(depth_image)

            # Fenster neu zeichnen
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Die Original-Funktion zur Steuerung des Agenten aufrufen
            processFrame(frame.pixels)
            agent_host.sendCommand( "turn " + str(current_yaw_delta_from_depth) )
        
        if not world_state.is_mission_running:
            break

    logger.info("Mission has stopped.")
    time.sleep(1)

# NEU: Fenster am Ende des Skripts schließen
plt.close()
logger.info("Done.")
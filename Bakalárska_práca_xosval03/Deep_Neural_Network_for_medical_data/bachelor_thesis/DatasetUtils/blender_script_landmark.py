import bpy
from math import *
from mathutils import *
import os
import time

teeth_mask = 'teeth-mask'
landmark_mask = 'landmark-mask'
input = 'input'
output = 'output'
output_mask = 'output-mask'

types_of_teeth = ['A_left', 'B_left', 'C_left', 'D_left', 'E_left', 'F_left', 'G_left', 'H_left']
types_of_teeth = ['A_right', 'B_right', 'C_right', 'D_right', 'E_right', 'F_right', 'G_right', 'H_left']
selected_teeth = types_of_teeth[6]  # L

targetName = 'model1'
# set your own target here
path = 'D:\\Renders\\RenderAnnotationVer30'
target = bpy.data.objects[targetName]
cam = bpy.data.objects['Camera']
t_loc_x = target.location.x
t_loc_y = target.location.y
t_loc_z = target.location.z

my_areas = bpy.context.workspace.screens[0].areas


def setLights(light='STUDIO'):
    for area in my_areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.light = light


target.rotation_euler[0] = 0
target.rotation_euler[1] = 0
target.rotation_euler[2] = 0
step = 5

angle = -35  # how many rotation steps
for x in range(angle, abs(angle), step):
    for y in range(angle, abs(angle), step):
        for z in range(angle, abs(angle), step):
            target.rotation_euler[0] = radians(x)
            target.rotation_euler[1] = radians(y)
            target.rotation_euler[2] = radians(z)
            bpy.ops.object.mode_set(mode='SCULPT')
            setLights()

            # bpy.types.View3DShading.light = 'STUDIO';

            file = os.path.join(path, landmark_mask, targetName, selected_teeth, input,
                                targetName + ('x' + str(x) + 'y' + str(y) + 'z' + str(z)))
            bpy.context.scene.render.filepath = file
            bpy.ops.render.opengl(write_still=True)
            bpy.ops.render.opengl(write_still=True)

            bpy.ops.object.mode_set(mode='VERTEX_PAINT')
            time.sleep(0.025)

            """
            file = os.path.join(path, targetName, selected_teeth, output ,targetName+
                                ('x=' + str(x) + 'y=' + str(y) + 'z=' + str(z)))

            bpy.context.scene.render.filepath = file
            bpy.ops.render.opengl(write_still=True)
            """
            setLights('FLAT')

            file = os.path.join(path, landmark_mask, targetName, selected_teeth, output_mask, targetName +
                                ('x' + str(x) + 'y' + str(y) + 'z' + str(z)))
            bpy.context.scene.render.filepath = file
            bpy.ops.render.opengl(write_still=True)
            bpy.ops.render.opengl(write_still=True)
            time.sleep(0.025)
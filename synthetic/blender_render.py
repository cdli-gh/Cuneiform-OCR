#Code for generating synthetic render images using blender
#input: 3d model + b/w sign_template image + tablet texture image
import bpy
import random
import os
#change model name here
o  = bpy.data.objects['Creativemachineslab-Tablet-1']
t  = o.active_material.node_tree

t_list = []
tex_list = []

#give b/w template input here
for r, d, f in os.walk("D:\\Folders\\CDLI_GSOC\\Dataset\\3d_files\\synthetic_3d_cuneiform\\final_templates\\760x890\\"):
    for file in f:
        t_list.append(file)

#give tablet texture input here
for r, d, f in os.walk("D:\\Folders\\CDLI_GSOC\\Dataset\\test_image_dataset\\temp_textures\\"):
    for file in f:
        tex_list.append(file)

for temp in t_list:      
    bpy.data.images.load("D:\\Folders\\CDLI_GSOC\\Dataset\\3d_files\\synthetic_3d_cuneiform\\final_templates\\760x890\\" + temp)

for tex in tex_list:      
    bpy.data.images.load("D:\\Folders\\CDLI_GSOC\\Dataset\\test_image_dataset\\temp_textures\\" + tex)

#for new models change the label name for the image input nodes and color ramp to respective names 
#give below
for template in t_list:
    image_name = random.sample(tex_list, k=1)[0]
    for node in o.active_material.node_tree.nodes:
        if node.bl_idname == 'ShaderNodeTexImage':
            if node.name == 'tablet_color' or node.name == 'tablet_roughness' or node.name == 'tablet_texh':
                node.image = bpy.data.images[image_name]
            elif node.name == 'tablet_template':
                node.image = bpy.data.images[template]
        elif node.bl_idname == 'ShaderNodeValToRGB':
            if node.name == 'ColorRamp_texheight':
                position_r = random.uniform(0.5,1.0)
                node.color_ramp.elements[1].position = round(position_r, 4)
                rgb_value = round(random.uniform(0.00,0.12), 4)
                node.color_ramp.elements[0].color = (rgb_value, rgb_value,rgb_value,1)
                rgb_value = round(random.uniform(0.5,0.8), 4)
                node.color_ramp.elements[1].color = (rgb_value, rgb_value,rgb_value,1)

    bpy.data.scenes['Scene'].render.filepath = 'D:\\Folders\\CDLI_GSOC\\Dataset\\3d_files\\synthetic_3d_cuneiform\\ouput\\760x890\\' + template
    bpy.ops.render.render(write_still=True)

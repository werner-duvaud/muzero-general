from datetime import datetime
import os
import IPython
e = IPython.embed


def generate_and_save_xml_file(weights, action_mode, is_eval):
  assert action_mode in ["joint_position", "joint_delta_position", "torque"]

  xml_str = generate_xml_str(weights, action_mode)

  now = datetime.now()
  time_stamp_str = now.strftime("%d_%m_%H_%M_%S")
  file_name = f"{len(weights)}_mugs_sawyer_shelf_{time_stamp_str}.xml"
  if is_eval:
    file_name = "eval_" + file_name
  else:
    file_name = "train_" + file_name
  destination_dir = os.path.dirname(os.path.abspath(__file__)) # same as this file's
  full_path = os.path.join(destination_dir, file_name)
  open(full_path, "w").write(xml_str)

  print("xml saved in ", full_path)
  return full_path


def generate_xml_str(weights, action_mode):
  weights = [w[0] for w in weights] # not the task has target pos in
  all_mugs = ""
  for i, w in enumerate(weights):
    one_mug = f"""
          <body name="mug{i}" pos="0 {-0.1*i - 0.4} 0.007">
              <joint name="mugjoint{i}" type="free" limited='false' damping="1" armature="0"/>
              <inertial pos="0 0 0" mass="{w}" diaginertia=".1 .1 .1"/>
              <geom name="mugGeom{i}"
                    mesh="cup_mesh"
                    conaffinity="1"
                    contype="1"
                    friction=".1 .005 .0001"
                    density="1384.7"
                    type="mesh"
                    material="white"
                    solimp="0.99 0.99 0.01"
                    solref="0.01 1"
                    rgba="1 1 1 1"/>
              <site name="mugSite{i}" pos="0 0 0" size="0.01" rgba="1 0 0 0"/>
              <site name="mugSite{i}_top" pos="0 0 0.1" size="0.01" rgba="1 0 0 0"/>
          </body>
    """
    all_mugs += one_mug


  joint_qpos = "-0.029997 -0.56 0.0299998 2.10036 0.11904 -1.16064 -1.46072 0.0092 -0.00705"
  all_mug_keys = ""
  for i, w in enumerate(weights):
    mug_i_key = f"0 {-0.1*i - 0.4} 0.007 1 0 0 0 " # the proper place they should stay
    all_mug_keys += mug_i_key
  all_mug_keys = all_mug_keys[:-1] # delete last space

  all_keys = f"""
      <keyframe>
          <key qpos='{joint_qpos} {all_mug_keys}'/>
      </keyframe>"""

  print(f"\nCONTROL MODE {action_mode}\n")
  if action_mode == "torque":
    return PART1_torque + all_mugs + PART2 + all_keys + PART3
  elif action_mode in ["joint_position", "joint_delta_position"]:
    return PART1_pos + all_mugs + PART2 + all_keys + PART3


PART1_torque = """
<?xml version="1.0" encoding="utf-8"?>
  <mujoco>
      <compiler meshdir="meshes" texturedir="textures"/>
      <include file="include/assets.xml"/>
      <include file="include/actuators_torqueCtrl.xml"/>
      <option timestep='0.0025'/>
  
  
      <worldbody>
          <light diffuse=".5 .5 .5" pos="0 -1 3" dir="0 0.5 -1"/> 
          <body name="sawyer">
              <include file="include/sawyer_chain.xml"/>
          </body>
  
          <include file="include/table.xml"/>
  
          <camera name="track" pos="0 -0.4 1.3" mode="fixed" euler="0.85 -0 0"/>
          <camera name="track_aux_shelf" pos="1.6 -0.4 1.7" mode="fixed" euler="0.95 0 1"/>
          <camera name="track_aux_shelf2" pos="1.4 0.4 0.6" mode="fixed" euler="1.3 0 1.57"/> 
          <camera name="look_at_mugs" pos="0.6 -0.5 0.3" mode="fixed" euler="1.3 0 1.57"/>

          
          """

PART1_pos = """
<?xml version="1.0" encoding="utf-8"?>
  <mujoco>
      <compiler meshdir="meshes" texturedir="textures"/>
      <include file="include/assets.xml"/>
      <include file="include/actuators_posCtrl.xml"/>
      <option timestep='0.0025'/>


      <worldbody>
          <light diffuse=".5 .5 .5" pos="0 -1 3" dir="0 0.5 -1"/> 
          <body name="sawyer">
              <include file="include/sawyer_chain.xml"/>
          </body>

          <include file="include/table.xml"/>

          <camera name="track" pos="0 -0.4 1.3" mode="fixed" euler="0.85 -0 0"/>
          <camera name="track_aux_shelf" pos="1.6 -0.4 1.7" mode="fixed" euler="0.95 0 1"/>
          <camera name="track_aux_shelf2" pos="1.4 0.4 0.6" mode="fixed" euler="1.3 0 1.57"/> 

          """


PART2 = """

        <body name="shelf_bowl1" pos="-0.44 1 0.042">
            <geom name="shelf_bowl_geom1"
                  mesh="bowl1_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="blue"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="0.8 0.8 0.8 0.4"/>
        </body>



        <body name="shelf_bowl2" pos="-0.23 0.99 0.042">
            <geom name="shelf_bowl_geom2"
                  mesh="bowl2_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="1 1 1 1"/>
        </body>


        <body name="shelf_bowl3" pos="0.1 1.0 0.042">
            <geom name="shelf_bowl_geom3"
                  mesh="bowl3_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="wood_material"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="0.6 0.6 0.6 1"/>
        </body>



        <body name="shelf_plate" pos="0.35 1.0 0.04">
            <geom name="shelf_plate_geom"
                  mesh="plate_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 0"
                  rgba="1 1 1 1"/>
        </body>


        <body name="shelf_plate_1" pos="0.35 1.0 0.06">
            <geom name="shelf_plate_geom_1"
                  mesh="plate_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 0"
                  rgba="1 1 1 1"/>
        </body>


        <body name="shelf_plate_2" pos="0.35 1.0 0.08">
            <geom name="shelf_plate_geom_2"
                  mesh="plate_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 0"
                  rgba="1 1 1 1"/>
        </body>

        <body name="shelf_plate_3" pos="0.35 1.0 0.10">
            <geom name="shelf_plate_geom_3"
                  mesh="plate_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 0"
                  rgba="1 1 1 1"/>
        </body>

        <body name="shelf_plate_4" pos="0.35 1.0 0.12">
            <geom name="shelf_plate_geom_4"
                  mesh="plate_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 0"
                  rgba="1 1 1 1"/>
        </body>


        <body name="shelf_bowl4" pos="0.26 1.2 0.215">
            <geom name="shelf_bowl_geom4"
                  mesh="bowl4_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="1 1 1 1"/>
        </body>

        <body name="shelf_cup2" pos="0.49 1.22 0.215">
            <geom name="shelf_cup_geom2"
                  mesh="cup2_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 0"
                  rgba="1 1 1 1"/>
        </body>


        <body name="shelf_cup21" pos="0.505 1.12 0.215">
            <geom name="shelf_cup_geom21"
                  mesh="cup2_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 3"
                  rgba="1 1 1 1"/>
        </body>


        <body name="shelf_cup22" pos="0.39 1.18 0.215">
            <geom name="shelf_cup_geom22"
                  mesh="cup2_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 1"
                  rgba="1 1 1 1"/>
        </body>



        <body name="shelf_cup23" pos="0.5 1.0 0.215">
            <geom name="shelf_cup_geom23"
                  mesh="cup2_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 -1"
                  rgba="1 1 1 1"/>
        </body>



        <body name="shelf_cup24" pos="0.4 1.05 0.215">
            <geom name="shelf_cup_geom24"
                  mesh="cup2_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="white"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  euler="1.58 0 -2"
                  rgba="1 1 1 1"/>
        </body>



        <body name="shelf_can1" pos="-0.526 1.2 0.215">
            <geom name="shelf_can_geom1"
                  mesh="can_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="red"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="0.8 0.8 0.8 1"/>
        </body>


        <body name="shelf_can2" pos="-0.53 0.9 0.215">
            <geom name="shelf_can_geom2"
                  mesh="can_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="blue"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="0.8 0.8 0.8 1"/>
        </body>


        <body name="shelf_box1" pos="-0.5 1.0 0.215">
            <geom name="shelf_box_geom1"
                  mesh="box_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="navy_blue"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="1 1 0.8 0.8"/>
        </body>


        <body name="shelf_box2" pos="-0.45 1.2 0.215">
            <geom name="shelf_box_geom2"
                  mesh="box_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="navy_blue"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="0.8 0.8 0.8 1"/>
        </body>


        <body name="shelf_bottle" pos="-0.44 1.09 0.215">
            <geom name="shelf_bottle_geom"
                  mesh="bottle_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="grey"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="1 1 1 1"/>
        </body>


        <body name="shelf_bottle1" pos="-0.52 1.1 0.215">
            <geom name="shelf_bottle_geom1"
                  mesh="bottle_mesh"
                  conaffinity="1"
                  contype="1"
                  friction=".1 .005 .0001"
                  density="1384.7"
                  type="mesh"
                  material="grey"
                  solimp="0.99 0.99 0.01"
                  solref="0.01 1"
                  rgba="0.8 0.8 0.8 1"/>
        </body>




        <body name="shelf" pos="0. 1.05 0.001">
            <geom type="box" contype="1" material="light_wood" size="0.6 0.2 0.007" name="level1" conaffinity="1"
                  pos="0 0 0.03" mass="1000"/>
            <geom type="box" contype="1" material="light_wood" size="0.6 0.2 0.007" name="level2" conaffinity="1"
                  pos="0 0 0.2" mass="1000"/>
            <geom type="box" contype="1" material="light_wood" size="0.61 0.2 0.007" name="cover" conaffinity="1"
                  pos="0 0 0.6" mass="1000"/>
<!--             <geom type="box" contype="1" material="light_wood" size="0.6 0.005 0.3" name="wall1" conaffinity="1"
                  pos="0.0 0.2 0.301" mass="1000"/> -->
            <!-- <geom rgba="0.3 0.3 1 1" type="box" solimp="0.99 0.99 0.01" contype="1" size="0.1 0.001 0.03" name="cover_wall2" conaffinity="1" pos="0.0 -0.1 0.034" mass="1000" solref="0.01 1"/> -->
            <geom material="light_wood" type="box" contype="1" size="0.005 0.2 0.3" name="wall3" conaffinity="1"
                  pos="0.6 0 0.301" mass="1000"/>
            <geom material="light_wood" type="box" contype="1" size="0.005 0.2 0.3" name="wall4" conaffinity="1"
                  pos="-0.6 0 0.301" mass="1000"/>
            <geom material="light_wood" type="box" contype="1" size="0.005 0.2 0.3" name="wall3_inner" conaffinity="1"
                  pos="0.575 0 0.301" mass="1000"/>
            <geom material="light_wood" type="box" contype="1" size="0.005 0.2 0.3" name="wall4_inner" conaffinity="1"
                  pos="-0.575 0 0.301" mass="1000"/>


            <site name="goal" pos="0 -0.1 0.22" size="0.02" rgba="0 0.8 0 0"/>
            <site name="goal_top" pos="0 -0.1 0.32" size="0.02" rgba="0 0.8 0 0"/>
            <!-- <joint type="slide" range="-0.2 0." axis="0 1 0" name="goal_slidey" pos="0 0 0" damping="1.0"/> -->
        </body>

    </worldbody>
        """

PART3 = """
  </mujoco>

"""

def test():
  w = [[1,10], [100000,2]]
  generate_and_save_xml_file(w, "torque", True)

if __name__ == '__main__':
  test()
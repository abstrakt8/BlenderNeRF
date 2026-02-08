import os
import shutil
import bpy
from . import helper, blender_nerf_operator


# global addon script variables
EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'

# camera on sphere operator class
class CameraOnSphere(blender_nerf_operator.BlenderNeRF_Operator):
    '''Camera on Sphere Operator'''
    bl_idname = 'object.camera_on_sphere'
    bl_label = 'Camera on Sphere COS'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='COS')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_data = self.get_camera_intrinsics(scene, camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='COS')
        if scene.splats and not scene.cos_all_frames: self.save_splats_ply(scene, output_path)

        # initial property might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath

        # other intial properties
        scene.init_sphere_exists = scene.show_sphere
        scene.init_camera_exists = scene.show_camera
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = camera

        if scene.test_data:
            # testing transforms
            output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TEST', method='COS')
            self.save_json(output_path, 'transforms_test.json', output_data)

        if scene.train_data:
            if not scene.show_camera: scene.show_camera = True

            # train camera on sphere
            sphere_camera = scene.objects[CAMERA_NAME]
            sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
            scene.camera = sphere_camera

            if scene.cos_all_frames:
                # 4DGS per-frame mode: one 3DGS dataset per animation frame
                frame_start = scene.frame_start
                frame_end = scene.frame_end
                total = (frame_end - frame_start + 1) * scene.cos_nb_frames
                count = 0

                for anim_frame in range(frame_start, frame_end + 1):
                    frame_dir = os.path.join(output_path, f"frame_{anim_frame:04d}")
                    os.makedirs(frame_dir, exist_ok=True)

                    # per-frame JSON (static 3DGS dataset frozen at this animation frame)
                    frame_output_data = dict(sphere_output_data)
                    frame_output_data['frames'] = self.get_camera_extrinsics_cos_perframe(scene, sphere_camera, anim_frame)
                    self.save_json(frame_dir, 'transforms_train.json', frame_output_data)

                    # per-frame splats
                    if scene.splats:
                        scene.frame_set(anim_frame)
                        self.save_splats_ply(scene, frame_dir)

                    # render stills for this animation frame
                    if scene.render_frames:
                        output_train = os.path.join(frame_dir, 'train')
                        os.makedirs(output_train, exist_ok=True)

                        scene.frame_set(anim_frame)
                        for cam_idx in range(scene.cos_nb_frames):
                            helper.sample_from_sphere(scene, camera_index=cam_idx)
                            scene.view_layers[0].update()

                            filename = f"r_{cam_idx:04d}"
                            scene.render.filepath = os.path.join(output_train, filename)
                            bpy.ops.render.render(write_still=True)

                            count += 1
                            print(f"BlenderNeRF 4DGS: Rendered {count}/{total}")

            else:
                # standard COS mode: freeze at frame_start, render stills per camera position
                sphere_output_data['frames'] = self.get_camera_extrinsics_cos(scene, sphere_camera)
                self.save_json(output_path, 'transforms_train.json', sphere_output_data)

                if scene.render_frames:
                    output_train = os.path.join(output_path, 'train')
                    os.makedirs(output_train, exist_ok=True)

                    scene.frame_set(scene.frame_start)
                    for cam_idx in range(scene.cos_nb_frames):
                        helper.sample_from_sphere(scene, camera_index=cam_idx)
                        scene.view_layers[0].update()

                        filename = f"r_{cam_idx:04d}"
                        scene.render.filepath = os.path.join(output_train, filename)
                        bpy.ops.render.render(write_still=True)

                        print(f"BlenderNeRF COS: Rendered {cam_idx + 1}/{scene.cos_nb_frames}")

        # cleanup (runs inline since we no longer use animation rendering)
        scene.render.filepath = scene.init_output_path

        # reset camera settings
        if not scene.init_camera_exists: helper.delete_camera(scene, CAMERA_NAME)
        if not scene.init_sphere_exists:
            objects = bpy.data.objects
            objects.remove(objects[EMPTY_NAME], do_unlink=True)
            scene.show_sphere = False
            scene.sphere_exists = False

        scene.camera = scene.init_active_camera

        # compress dataset and remove folder (only keep zip)
        shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
        shutil.rmtree(output_path)

        return {'FINISHED'}

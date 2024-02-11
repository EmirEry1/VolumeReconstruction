import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

focal_length_mm = 8.2

camera_distance_mm = 420
sensor_width_mm = 11.3
sensor_height_mm = 7.1

mm_to_pixel_coefficient = 1920/11.3

focal_distance_to_origin_mm = camera_distance_mm - focal_length_mm

camera_0_focal_coordinate_mm = [0, focal_distance_to_origin_mm, 0]
camera_1_focal_coordinate_mm = [focal_distance_to_origin_mm, 0, 0]
camera_2_focal_coordinate_mm = [-focal_distance_to_origin_mm, 0, 0]
camera_3_focal_coordinate_mm = [0, -focal_distance_to_origin_mm, 0]


def find_projection_on_sensor(world_focal_coordinate_mm, world_pixel_location_mm):
    #With this if condition I determine which camera the focal point in the parameter corresonds to
    
    if world_focal_coordinate_mm[1] == focal_distance_to_origin_mm:
        #Camera 0
        #This matrix is used to convert the point to camera coordinates
        camera_0_conversion_matrix = [[0, 1, 0, -focal_distance_to_origin_mm], [-1, 0, 0, 0], [0, 0, 1, 0]]
        camera_0_conversion_matrix = np.array(camera_0_conversion_matrix)
        pixel_camera_location_mm = np.dot(camera_0_conversion_matrix, np.append(world_pixel_location_mm, 1))
    elif world_focal_coordinate_mm[0] == focal_distance_to_origin_mm:
        #Camera 1

        #This matrix is used to convert the point to camera coordinates
        camera_1_conversion_matrix = [[1, 0, 0, -focal_distance_to_origin_mm], [0, 1, 0, 0], [0, 0, 1, 0]]
        pixel_camera_location_mm = np.dot(camera_1_conversion_matrix, np.append(world_pixel_location_mm, 1))

    elif world_focal_coordinate_mm[0] == -focal_distance_to_origin_mm:
        #Camera 2

        #This matrix is used to convert the point to camera coordinates
        camera_2_conversion_matrix = [[-1, 0, 0, -focal_distance_to_origin_mm], [0, -1, 0, 0], [0, 0, 1, 0]]
        pixel_camera_location_mm = np.dot(camera_2_conversion_matrix, np.append(world_pixel_location_mm, 1))

    elif world_focal_coordinate_mm[1] == -focal_distance_to_origin_mm:
        #Camera 3

        #This matrix is used to convert the point to camera coordinates
        camera_3_conversion_matrix = [[0, -1, 0, -focal_distance_to_origin_mm], [1, 0, 0, 0], [0, 0, 1, 0]]
        pixel_camera_location_mm = np.dot(camera_3_conversion_matrix, np.append(world_pixel_location_mm, 1))
    similarity_ratio = focal_length_mm/pixel_camera_location_mm[0]
  
    
    image_plane_conversion_matrix = [[0, 0, similarity_ratio*mm_to_pixel_coefficient, 960], [0, similarity_ratio*mm_to_pixel_coefficient, 0, 600]]
    image_plane_conversion_matrix = np.array(image_plane_conversion_matrix)
    sensor_projection = np.round(np.dot(image_plane_conversion_matrix, np.append(pixel_camera_location_mm, 1)))
    return sensor_projection.astype(int)

camera_images = []
camera_0_image = cv2.imread("fish-1/frame9_cam0_msk.jpg")
camera_1_image = cv2.imread("fish-1/frame9_cam1_msk.jpg")
camera_2_image = cv2.imread("fish-1/frame9_cam2_msk.jpg")
camera_3_image = cv2.imread("fish-1/frame9_cam3_msk.jpg")  

camera_images.append([camera_0_image, camera_1_image, camera_2_image, camera_3_image])

camera_0_image = cv2.imread("fish-2/frame12_cam0_msk.jpg")
camera_1_image = cv2.imread("fish-2/frame12_cam1_msk.jpg")
camera_2_image = cv2.imread("fish-2/frame12_cam2_msk.jpg")
camera_3_image = cv2.imread("fish-2/frame12_cam3_msk.jpg")  

camera_images.append([camera_0_image, camera_1_image, camera_2_image, camera_3_image])

camera_0_image = cv2.imread("fish-3/frame12_cam0_msk.jpg")
camera_1_image = cv2.imread("fish-3/frame12_cam1_msk.jpg")
camera_2_image = cv2.imread("fish-3/frame12_cam2_msk.jpg")
camera_3_image = cv2.imread("fish-3/frame12_cam3_msk.jpg")  

camera_images.append([camera_0_image, camera_1_image, camera_2_image, camera_3_image])



voxal_size_mm = [10.5, 10.5, 10.5]
total_grid_size_mm = [840, 840, 840] 

volumes = []

for fish in camera_images:
    camera_0_image = fish[0]
    camera_1_image = fish[1]
    camera_2_image = fish[2]
    camera_3_image = fish[3]

    positively_voted_voxels = []
    negatively_voted_voxels = []

    range_values = np.arange(-409.5, 410.0, 10.5)
    voxel_x, voxel_y, voxel_z = np.meshgrid(range_values, range_values, range_values)
    voxels = np.column_stack((voxel_x.ravel(), voxel_y.ravel(), voxel_z.ravel()))   
    
    for voxel_mm in voxels:
        camera_0_projection_pixel = find_projection_on_sensor(camera_0_focal_coordinate_mm, voxel_mm)
        camera_1_projection_pixel = find_projection_on_sensor(camera_1_focal_coordinate_mm, voxel_mm)        
        camera_2_projection_pixel = find_projection_on_sensor(camera_2_focal_coordinate_mm, voxel_mm)        
        camera_3_projection_pixel = find_projection_on_sensor(camera_3_focal_coordinate_mm, voxel_mm)  
        votes = 0
        if camera_0_projection_pixel[0] < 1920  and camera_0_projection_pixel[0] >= 0 and camera_0_projection_pixel[1] >= 0 and camera_0_projection_pixel[1] < 1200: 
            if camera_0_image[camera_0_projection_pixel[1]][camera_0_projection_pixel[0]][0] > 180:
                votes += 1
        if camera_1_projection_pixel[0] < 1920  and camera_1_projection_pixel[0] >= 0 and camera_1_projection_pixel[1] >= 0 and camera_1_projection_pixel[1] < 1200:
            if camera_1_image[camera_1_projection_pixel[1]][camera_1_projection_pixel[0]][0] > 180:
                votes += 1
        if camera_2_projection_pixel[0] < 1920  and camera_2_projection_pixel[0] >= 0 and camera_2_projection_pixel[1] >= 0 and camera_2_projection_pixel[1] < 1200:
            if camera_2_image[camera_2_projection_pixel[1]][camera_2_projection_pixel[0]][0] > 180:
                votes += 1
        if camera_3_projection_pixel[0] < 1920  and camera_3_projection_pixel[0] >= 0 and camera_3_projection_pixel[1] >= 0 and camera_3_projection_pixel[1] < 1200:
            if camera_3_image[camera_3_projection_pixel[1]][camera_3_projection_pixel[0]][0] > 180:
                votes += 1
        if votes >= 3:
            positively_voted_voxels.append(voxel_mm)
        else:
            negatively_voted_voxels.append(voxel_mm)

    positively_voted_voxels = np.array(positively_voted_voxels)
    negatively_voted_voxels = np.array(negatively_voted_voxels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(positively_voted_voxels[:, 0], positively_voted_voxels[:, 1], positively_voted_voxels[:, 2], c='r', marker='o', label='Marked Voxels')

    ax.grid(True)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    min_val = min(positively_voted_voxels.min(), negatively_voted_voxels.min())
    max_val = max(positively_voted_voxels.max(), negatively_voted_voxels.max())


    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)


    plt.title('Marked Voxels in 3D Space')

    plt.legend()
    plt.show()

    volume_mm3 = len(positively_voted_voxels)*(voxal_size_mm[0])*(voxal_size_mm[1])*(voxal_size_mm[2])
    volumes.append(volume_mm3)

print("Volume of Fish 1 (cm^3): " + str(volumes[0]/1000))
print("Volume of Fish 2 (cm^3): " + str(volumes[1]/1000))
print("Volume of Fish 3 (cm^3): " + str(volumes[2]/1000))

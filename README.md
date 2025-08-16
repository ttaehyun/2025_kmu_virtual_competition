# 2025_kmu_virtual_competition

roslaunch robot_setup_tf tf.launch

rosrun obstacle_detector roi_marker_publisher_node

rosrun obstacle_detector obstacle_extractor_node

rosrun obstacle_detector obstacle_points_base_link_pub


추후 수정사항 : roi 바깥 점에 대해 계산하지 않기.


import json
settings_filename = "settings.json"
settings = {}

# Input
settings["video_filename"] = "N2-2_2019-07-29-133409-0000-2.avi"
settings["start_frame"] = 0
settings["limit_images_to"] = 600
settings["fps"] = 20.0
settings["px_to_mm"] = 0.04
settings["darkfield"] = False
settings["stop_after_example_output"] = False

# Output
settings["save_as"] = "Demo-data"
settings["output_overlayed_images"] = 3
settings["font_size"] = 8
settings["fig_size"] = (20, 20)
settings["scale_bar_size"] = 1.0
settings["scale_bar_thickness"] = 7

# Z-filtering
settings["use_images"] = 100
settings["use_around"] = 5
settings["Z_skip_images"] = 1

# Thresholding
settings["keep_dead_method"] = False
settings["std_px"] = 64
settings["threshold"] = 9
settings["opening"] = 1
settings["closing"] = 3
settings["skeletonize"] = True
settings["prune_size"] = 0
settings["do_full_prune"] = True

# Locating
settings["min_size"] = 25
settings["max_size"] = 120
settings["minimum_ecc"] = 0.9

# Form trajectories
settings["max_dist_move"] = 10
settings["min_track_length"] = 50
settings["memory"] = 5

# Bending statistics
settings["bend_threshold"] = 2.1
settings["minimum_bends"] = 0.0

# Velocities
settings["frames_to_estimate_velocity"] = 3

# Dead worm statistics
settings["maximum_bpm"] = 0.5
settings["maximum_velocity"] = 0.1

# Regions
settings["regions"] = {'': {
    'y': [377.3678756476688,
          145.2435233160627,
          105.45077720207291,
          536.5388601036273,
          891.3575129533681,
          1471.6683937823836,
          1833.119170984456,
          1756.8497409326426,
          1498.1968911917102,
          1077.056994818653,
          682.4455958549227,
          519.9585492227982],
    'x': [274.0725388601038,
          675.316062176166,
          1205.8860103626944,
          1733.1398963730571,
          1845.8860103626944,
          1696.6632124352332,
          1222.4663212435232,
          615.6269430051814,
          293.9689119170987,
          118.21761658031107,
          78.42487046632152,
          184.53886010362714]}}

### Optimisation tools
settings["lower"] = 0
settings["upper"] = 100
settings["use_average"] = False
settings["cutoff_filter"] = True
settings["extra_filter"] = True
settings["Bends_max"] = 20.0
settings["Speed_max"] = 0.035

with open(settings_filename, "w") as f:
    json.dump(settings, f, indent=4)



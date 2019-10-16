### Input
filename = "C:/Users/Mandy/Downloads/JBKirkegaard-multiwormtracker-85a540a2a163/D1_swimming_2016-04-01-122434-0000.avi"
start_frame = 0
limit_images_to = 10
fps = 20.0
px_to_mm = 1.0
darkfield = False
stop_after_example_output = False

### Output
save_as = "C:/Users/Mandy/Downloads/JBKirkegaard-multiwormtracker-85a540a2a163/"
output_overlayed_images = 0
font_size =  8
fig_size = (20,20)
scale_bar_size = 1.0
scale_bar_thickness = 7

### Z-filtering
use_images = 100
use_around = 5
Z_skip_images = 1

### Thresholding
keep_dead_method = False
std_px = 64
threshold = 8
opening = 2
closing = 4
skeletonize = False
prune_size = 0
do_full_prune = False

### Locating
min_size = 50
max_size = 1000
minimum_ecc = 0.7

### Form trajectories
max_dist_move = 5
min_track_length = 5
memory = 5

### Bending statistics
bend_threshold = 2.0
minimum_bends = 0.0

### Velocities
frames_to_estimate_velocity = 3

### Dead worm statistics
maximum_bpm = 0.5
maximum_velocity = 0.1

### Regions
regions = {}

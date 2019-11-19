import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from multiprocessing import Pool, cpu_count
from scipy import interpolate, ndimage
import cv2
import os
import time
import mahotas as mh
import pandas as pd
import trackpy as tp
from skimage import measure, morphology, io
from math import factorial
import random
import skimage.draw
import pickle
import warnings
import matplotlib.path as mplPath
from collections import defaultdict, Counter
from skimage.transform import resize
import json
import traceback
from scipy.signal import savgol_filter
import functools


def run_tracker(settings):
    """Run the tracker with the given settings."""

    # Do some adjustments
    settings = settings.copy()
    settings["frames_to_estimate_velocity"] = min([
        settings["frames_to_estimate_velocity"],
        settings["min_track_length"]])
    settings["bend_threshold"] /= 100.

    video = Video(settings, grey=True)

    print('Video shape:', video[0].shape)

    regions = settings["regions"]
    try:
        len(regions)
    except Exception:
        regions = {}
    if len(regions) == 0:
        im = np.ones_like(video[0])
        all_regions = im > 0.1
    else:
        all_regions = np.zeros_like(video[0])
        for key, d in list(regions.items()):
            im = np.zeros_like(video[0])
            rr, cc = skimage.draw.polygon(np.array(d['y']), np.array(d['x']))
            try:
                im[rr, cc] = 1
            except IndexError:
                print('Region "', key, '" cannot be applied to video',
                      settings["video_filename"])
                print('Input image sizes do not match.')
                return
            all_regions += im
        all_regions = all_regions > 0.1
    settings["all_regions"] = all_regions
    settings["regions"] = regions

    t0 = time.time()
    save_folder = settings["save_as"]
    ims_folder = os.path.join(save_folder, 'imgs')
    if not os.path.exists(ims_folder):
        os.mkdir(ims_folder)

    # Analysis
    print_data, locations = track_all_locations(video, settings)

    if settings["stop_after_example_output"]:
        return print_data, None, None
    track = form_trajectories(locations, settings)

    results = extract_data(track, settings)
    if not check_for_worms(results["particles"], settings):
        print('No worms detected. Stopping!')
        return
    # Output
    write_results_file(results, settings)

    print('Done (in %.1f minutes).' % ((time.time() - t0) / 60.))
    video.release()
    return print_data, results

class Video:
    def __init__(self, settings, grey=False):
        video_filename = settings["video_filename"]
        if not os.path.exists(video_filename):
            raise RuntimeError(f"{video_filename} does not exist.")
        self.cap = cv2.VideoCapture(video_filename)
        self.fname = video_filename
        self.name = "".join(video_filename.split(".")[:-1]).replace('/', '_')
        self.len = (self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    - settings["start_frame"])
        self.start_frame = settings["start_frame"]
        limit_images_to = settings["limit_images_to"]
        if (limit_images_to
              and limit_images_to < (self.len - self.start_frame)):
            self.len = limit_images_to
        self.grey = grey
        if grey:
            for _ in range(100):
                ret, frame = self.cap.read()
                if ret:
                    break
            if len(frame.shape) == 2:
                self.grey = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __next__(self):
        ret = False
        for _ in range(100):
            ret, frame = self.cap.read()
            if ret:
                break
            time.sleep(0.1 * random.random())
        if ret:
            if self.grey:
                return frame[:, :, 0]
            else:
                return frame
        else:
            raise StopIteration

    def set_index(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    def restart(self):
        self.set_index(self.start_frame)

    def __getitem__(self, i):
        if i < 0:
            i += self.len
        self.set_index(self.start_frame + i)
        return next(self)

    def __len__(self):
        return int(self.len)

    def release(self):
        self.cap.release()


def track_all_locations(video, settings):
    """Track and get all locations."""
    def get_Z_brightness(zi):
        if settings["keep_dead_method"]:
            return find_Z_withdead(video, settings, *zi)
        else:
            return find_Z(video, settings, *zi)

    apply_indeces = list(
        map(int, list(np.linspace(0, len(video),
                                  len(video) // settings["use_images"] + 2))))
    apply_indeces = list(zip(apply_indeces[:-1], apply_indeces[1:]))
    Z_indeces = [(max([0, i - settings["use_around"]]),
                  min(j + settings["use_around"], len(video)))
                 for i, j in apply_indeces]

    # Get frames0 print material
    Z, mean_brightness = get_Z_brightness(Z_indeces[0])
    print_data = process_frame(settings, Z, mean_brightness,
                               len(video), (0, video[0]),return_plot=True)

    if settings["stop_after_example_output"]:
        return print_data, None

    # Process all frames
    args = list(zip(apply_indeces, Z_indeces))
    def locate(args):
        i, zi = args
        Z, mean_brightness = get_Z_brightness(zi)
        return process_frames(video, settings, *i, Z=Z,
                       mean_brightness=mean_brightness)
    split_results = list(map(locate, args))
    locations = []
    for l in split_results:
        locations += l
    return print_data, locations


def process_frame(settings, Z, mean_brightness, nframes, args,
                  return_plot=False):
    i, frameorig = args
    print(' : Locating in frame %i/%i'
          % (i + 1 + settings["start_frame"],
             nframes + settings["start_frame"]))

    if mean_brightness:
        frame = frameorig * mean_brightness / np.mean(frameorig)
    else:
        frame = np.array(frameorig, dtype=np.float64)
    frame = np.abs(frame - Z) * settings["all_regions"]
    if (frame > 1.1).any():
        frame /= 255.

    thresholded = frame > (settings["threshold"] / 255.)
    opening = settings["opening"]
    closing = settings["closing"]
    save_folder = settings["save_as"]
    if opening > 0:
        frame_after_open = ndimage.binary_opening(
            thresholded,
            structure=np.ones((opening, opening))).astype(np.int)
    else:
        frame_after_open = thresholded

    if closing > 0:
        frame_after_close = ndimage.binary_closing(
            frame_after_open,
            structure=np.ones((closing, closing))).astype(np.int)
    else:
        frame_after_close = frame_after_open

    labeled, _ = mh.label(frame_after_close, np.ones(
        (3, 3), bool))  # change here?
    sizes = mh.labeled.labeled_size(labeled)

    remove = np.where(np.logical_or(sizes < settings["min_size"],
                                    sizes > settings["max_size"]))
    labeled_removed = mh.labeled.remove_regions(labeled, remove)
    labeled_removed, n_left = mh.labeled.relabel(labeled_removed)

    props = measure.regionprops(labeled_removed, coordinates='xy')
    prop_list = [{"area": props[j].area, "centroid":props[j].centroid,
                  "eccentricity":props[j].eccentricity,
                  "area_eccentricity":props[j].eccentricity,
                  "minor_axis_length":props[j].minor_axis_length /
                  (props[j].major_axis_length + 0.001)}
                 for j in range(len(props))]
    if settings["skeletonize"]:
        skeletonized_frame = morphology.skeletonize(frame_after_close)
        skeletonized_frame = prune(skeletonized_frame,
                                   settings["prune_size"])

        skel_labeled = labeled_removed * skeletonized_frame
        if settings["do_full_prune"]:
            skel_labeled = prune_fully(skel_labeled)

        skel_props = measure.regionprops(skel_labeled, coordinates='xy')
        for j in range(len(skel_props)):
            prop_list[j]["length"] = skel_props[j].area
            prop_list[j]["eccentricity"] = skel_props[j].eccentricity
            prop_list[j]["minor_axis_length"] = \
                skel_props[j].minor_axis_length\
                / (skel_props[j].major_axis_length + 0.001)

    if return_plot:
        return (sizes, save_folder, frameorig, Z, frame, thresholded,
                frame_after_open, frame_after_close, labeled, labeled_removed,
                (skel_labeled if settings["skeletonize"] else None))

    output_overlayed_images = settings["output_overlayed_images"]
    if i < output_overlayed_images or output_overlayed_images is None:
        warnings.filterwarnings("ignore")
        io.imsave(os.path.join(save_folder, "imgs", '%05d.jpg' % (i)),
                  np.array(labeled_removed == 0, dtype=np.float32))
        warnings.filterwarnings("default")

    return prop_list


def process_frames(video, settings, i0, i1, Z, mean_brightness):
    """Frocess frames from i0 to i1."""
    func = functools.partial(
        process_frame, settings, Z, mean_brightness, len(video))
    def args():
        for i in range(i0, i1):
            yield i, video[i]

    if settings["parallel"]:
        p = Pool(cpu_count())
        results = p.imap(func, args())
    else:
        results = map(func, args())

    return results


def form_trajectories(loc, settings):
    global particles, P, T, bends, track
    print()
    print('Forming worm trajectories...', end=' ')
    data = {'x': [], 'y': [], 'frame': [],
            'eccentricity': [], 'area': [],
            'minor_axis_length': [],
            'area_eccentricity': []}
    for t, l in enumerate(loc):
        data['x'] += [d['centroid'][0] for d in l]
        data['y'] += [d['centroid'][1] for d in l]
        data['eccentricity'] += [d['eccentricity'] for d in l]
        data['area_eccentricity'] += [d['area_eccentricity'] for d in l]
        data['minor_axis_length'] += [d['minor_axis_length'] for d in l]
        data['area'] += [d['area'] for d in l]
        data['frame'] += [t] * len(l)
    data = pd.DataFrame(data)
    try:
        track = tp.link_df(data, search_range=settings["max_dist_move"],
                           memory=settings["memory"])
    except tp.linking.SubnetOversizeException:
        raise RuntimeError(
            'Linking problem too complex.'
            ' Reduce maximum move distance or memory.')
    track = tp.filter_stubs(track, min([settings["min_track_length"],
                                        len(loc)]))
    try:
        with open(os.path.join(settings["save_as"], 'track.p'), 'bw') as trackfile:
            pickle.dump(track, trackfile)
    except Exception:
        traceback.print_exc()
        print('Warning: no track file saved. Track too long.')
        print('         plot_path.py will not work on this file.')

    return track


def extract_data(track, settings):
    """Extract data from track and return a pandas DataFrame."""
    P = track['particle']
    particles = list(set(P))
    T = track['frame']
    X = track['x']
    Y = track['y']
    bends = []
    velocites = []
    max_speed = []
    areas = []
    eccentricity = []
    region = []
    move_per_bends = []
    region_particles = defaultdict(list)
    round_ratio = []

    regions = settings["regions"]
    if len(regions) > 1:
        reg_paths = make_region_paths(regions)

    # Iterate reversed to allow deletion
    for pi, p in reversed(list(enumerate(particles))):
        # Define signals
        t = T[P == p]
        ecc = track['eccentricity'][P == p]
        area_ecc = track['area_eccentricity'][P == p]
        # mal = track['minor_axis_length'][P == p]
        area = track['area'][P == p]

        # Smooth bend signal
        x = np.arange(min(t), max(t) + 1)
        f = interpolate.interp1d(t, ecc)
        y = f(x)
        smooth_y = savgol_filter(y, 7, 2)

        # Use eccentricity of non-skeletonized to filter worm-like
        f = interpolate.interp1d(t, area_ecc)
        y = f(x)
        area_ecc = savgol_filter(y, 7, 2)

        # Interpolate circle-like worms
        # (these are removed later if count is low)
        idx = area_ecc > settings["minimum_ecc"]
        if sum(idx) > 0:
            smooth_y = np.interp(x, x[idx], smooth_y[idx])
            roundness = 1.0 - float(sum(idx)) / float(len(idx))
            round_ratio.append(roundness)
        else:
            lengthX = 0.001 / len(idx)
            smooth_y = np.arange(0.991, 0.992, lengthX)
            np.random.shuffle(smooth_y)
            roundness = 1.0 - float(sum(idx)) / float(len(idx))
            round_ratio.append(roundness)

        # Bends
        bend_times = extract_bends(x, smooth_y, settings)
        if len(bend_times) < settings["minimum_bends"]:
            del particles[pi]
            continue
        bl = form_bend_array(bend_times, T[P == p])
        if len(bl) > 0:
            bends.append(np.array(bl) * 1.0)
        else:
            bends.append(np.array([0.0] * len(T[P == p])))

        px_to_mm = settings["px_to_mm"]
        # Area
        if settings["skeletonize"]:
            areas.append(np.median(area) * px_to_mm)
        else:
            areas.append(np.median(area) * px_to_mm**2)

        # Eccentricity
        eccentricity.append(np.mean(area_ecc))

        # Velocity
        velocity = extract_velocity(T[P == p], X[P == p], Y[P == p], settings)
        velocites.append(velocity)

        # Max velocity: 90th percentile to avoid skewed results due to tracking
        # inefficiency
        max_speeds = extract_max_speed(T[P == p], X[P == p], Y[P == p],
                                       settings)
        max_speed.append(max_speeds)

        # Move per bend
        move_per_bend = extract_move_per_bend(
            bends[-1], T[P == p], X[P == p], Y[P == p], px_to_mm)
        move_per_bends.append(move_per_bend)

    # Appended lists need to be reversed to same order as particles
    bends, velocites, areas, \
        max_speed, move_per_bends, round_ratio, eccentricity = [
            list(reversed(x)) for x in [
                bends, velocites, areas,
                max_speed, move_per_bends, round_ratio, eccentricity]]

    # Sort out low bend number particles
    for i in reversed(list(range(len(bends)))):
        if bends[i][-1] < settings["minimum_bends"]:
            del bends[i]
            del particles[i]
            del velocites[i]
            del areas[i]
            del eccentricity[i]
            del move_per_bends[i]
            del max_speed[i]
            del round_ratio[i]
            del eccentricity[i]

    # BPM
    bpm = []  # beats per minute
    bendsinmovie = []
    appears_in = []
    fps = settings["fps"]
    for i, p in enumerate(particles):
        bpm.append(bends[i][-1] / np.ptp(T[P == p]) * 60 * fps)
        x = (settings["limit_images_to"] / fps)
        bendsinmovie.append(
            bends[i][-1] / np.ptp(T[P == p]) * x * fps)  # CHANGE
        appears_in.append(len(bends[i]))

    # Cut off-tool for skewed statistics
    if settings["cutoff_filter"]:
        list_number = []
        frames = []
        for t in set(T):
            if t >= settings["lower"] and t <= settings["upper"]:
                particles_present = len(set(P[T == t]))
                frames.append(t)
                list_number.append(particles_present)

        list_number = np.array(list_number)

        if settings["use_average"]:
            cut_off = int(np.sum(list_number) / len(list_number)) + \
                (np.sum(list_number) % len(list_number) > 0)
        else:
            cut_off = max(list_number)

        # cut off based on selected frames
        bends = bends[:cut_off]
        original_particles = len(particles)
        velocites = velocites[:cut_off]
        areas = areas[:cut_off]
        bpm = bpm[:cut_off]
        bendsinmovie = bendsinmovie[:cut_off]
        move_per_bends = move_per_bends[:cut_off]
        appears_in = appears_in[:cut_off]
        max_speed = max_speed[:cut_off]
        particles = particles[:cut_off]
        round_ratio = round_ratio[:cut_off]
        eccentricity = eccentricity[:cut_off]

    else:
        original_particles = len(particles)
        list_number = 0
        frames = 0

    # Cut off-tool for boundaries (spurious worms)
    spurious_worms = 0
    if settings["extra_filter"]:
        for i in reversed(list(range(len(bends)))):
            if (bpm[i] > settings["Bends_max"]
                  and velocites[i] < settings["Speed_max"]):
                del bends[i]
                del particles[i]
                del velocites[i]
                del areas[i]
                del move_per_bends[i]
                del max_speed[i]
                del bpm[i]
                del bendsinmovie[i]
                del appears_in[i]
                del round_ratio[i]
                del eccentricity[i]
                spurious_worms += 1
    else:
        spurious_worms = 0

    for pi, p in list(enumerate(particles)):
        # Indetify region
        if len(regions) > 1:
            this_reg = identify_region(X[P == p], Y[P == p], reg_paths)
            if this_reg is None:
                continue
        else:
            this_reg = 'all'
        region.append(this_reg)
        region_particles[this_reg].append(p)

    results = dict(
        region=region,
        region_particles=region_particles,
        bends=bends,
        particles=particles,
        velocites=velocites,
        areas=areas,
        move_per_bends=move_per_bends,
        bpm=bpm,
        bendsinmovie=bendsinmovie,
        appears_in=appears_in,
        max_speed=max_speed,
        spurious_worms=spurious_worms,
        original_particles=original_particles,
        list_number=list_number,
        frames=frames,
        round_ratio=round_ratio,
        eccentricity=eccentricity)
    for key in results:
        results[key] = np.asarray(results[key])
    results["track"] = track
    return results


# =============================================================================
# --- Utilities Functions ---
# =============================================================================

def find_Z(video, settings, i0, i1):
    # Adjust brightness:
    frame = video[(i0 + i1) // 2]
    mean_brightness = np.mean(frame)
    if mean_brightness > 1:
        mean_brightness /= 255.
    Z = np.zeros_like(frame, dtype=np.float64)
    if settings["darkfield"]:
        minv = np.zeros_like(frame, dtype=np.float64) + 256
    else:
        minv = np.zeros_like(frame, dtype=np.float64) - 256
    for i in range(i0, i1, settings["Z_skip_images"]):
        frame = video[i]
        frame = frame * mean_brightness / np.mean(frame)
        diff = frame
        if settings["darkfield"]:
            logger = diff < minv
        else:
            logger = diff > minv
        minv[logger] = diff[logger]
        Z[logger] = frame[logger]
    return Z, mean_brightness


def find_Z_withdead(video, settings, i0, i1):
    frame = video[(i0 + i1) // 2]
    Y, X = np.meshgrid(np.arange(frame.shape[1]),
                       np.arange(frame.shape[0]))
    thres = cv2.adaptiveThreshold(
        frame, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 2 * (settings["std_px"] // 2) + 1, 0) < 0.5
    mask = np.logical_not(thres)
    vals = frame[mask]
    x = X[mask]
    y = Y[mask]
    Z = interpolate.griddata((x, y), vals, (X, Y), method='nearest')
    return Z, False


def find_skel_endpoints(skel):
    skel_endpoints = [
        np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]]),
        np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]]),
        np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]]),
        np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]]),
        np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]]),
        np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]]),
        np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]]),
        np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])]

    ep = 0
    for skel_endpoint in skel_endpoints:
        ep += mh.morph.hitmiss(skel, skel_endpoint)

    return ep


def prune(skel, size):
    for _ in range(size):
        endpoints = find_skel_endpoints(skel)
        skel = np.logical_and(skel, np.logical_not(endpoints))
    return skel


def prune_fully(skel_labeled):
    for k in range(1000):
        endpoints = find_skel_endpoints(skel_labeled > 0) > 0
        idx = np.argwhere(endpoints)
        reg = skel_labeled[idx[:, 0], idx[:, 1]]
        count = Counter(reg)
        idx = np.array([idx[i, :] for i in range(len(reg))
                        if count[reg[i]] > 2])
        if len(idx) == 0:
            break
        endpoints[:] = 1
        endpoints[idx[:, 0], idx[:, 1]] = 0
        skel_labeled *= endpoints
    return skel_labeled


def check_for_worms(particles, settings):
    if len(particles) == 0:
        with open(os.path.join(settings["save_as"], 'results.txt'), 'w') as f:
            f.write('---------------------------------\n')
            f.write('    Results for %s \n' % settings["video_filename"])
            f.write('---------------------------------\n\n')
            f.write('No worms detected. Check your settings.\n\n')
        return False
    return True


def make_region_paths(regions):
    reg_paths = {}
    for key, d in list(regions.items()):
        reg_paths[key] = mplPath.Path(
            np.array(list(zip(d['x'] + [d['x'][0]], d['y'] + [d['y'][0]]))))
    return reg_paths


def identify_region(xs, ys, reg_paths):
    for x, y in zip(xs, ys):
        for key, path in list(reg_paths.items()):
            if path.contains_point((y, x)):
                return key
    return None


def extract_bends(x, smooth_y, settings):
    # Find extrema
    ex = (np.diff(np.sign(np.diff(smooth_y))).nonzero()[0] + 1)
    if len(ex) >= 2 and ex[0] == 0:
        ex = ex[1:]
    bend_times = x[ex]
    bend_magnitude = smooth_y[ex]

    # Sort for extrema satisfying criteria
    idx = np.ones(len(bend_times))
    index = 1
    prev_index = 0
    while index < len(bend_magnitude):
        dist = abs(bend_magnitude[index] - bend_magnitude[prev_index])
        if dist < settings["bend_threshold"]:
            idx[index] = 0
            if index < len(bend_magnitude) - 1:
                idx[index + 1] = 0
            index += 2  # look for next maximum/minimum (not just extrema)
        else:
            prev_index = index
            index += 1
    bend_times = bend_times[idx == 1]
    return bend_times


def form_bend_array(bend_times, t_p):
    bend_i = 0
    bl = []
    if len(bend_times):
        for i, t in enumerate(t_p):
            if t > bend_times[bend_i]:
                if bend_i < len(bend_times) - 1:
                    bend_i += 1
            bl.append(bend_i)
    return bl


def extract_velocity(tt, xx, yy, settings):
    ftev = settings["frames_to_estimate_velocity"]
    dtt = -(np.roll(tt, ftev) - tt)[ftev:]
    dxx = (np.roll(xx, ftev) - xx)[ftev:]
    dyy = (np.roll(yy, ftev) - yy)[ftev:]
    velocity = (settings["px_to_mm"] * settings["fps"]
                * np.median(np.sqrt(dxx**2 + dyy**2) / dtt))
    return velocity


def extract_max_speed(tt, xx, yy, settings):
    ftev = settings["frames_to_estimate_velocity"]
    dtt = -(np.roll(tt, ftev) - tt)[ftev:]
    dxx = (np.roll(xx, ftev) - xx)[ftev:]
    dyy = (np.roll(yy, ftev) - yy)[ftev:]
    percentile = (
        settings["px_to_mm"] * settings["fps"] *
        np.percentile((np.sqrt(dxx**2 + dyy**2) / dtt), 90))
    return percentile


def extract_move_per_bend(bl, tt, xx, yy, px_to_mm):
    bend_i = 1
    j = 0
    dists = []
    for i in range(len(bl)):
        if int(bl[i]) == bend_i:
            xi = np.interp(i, tt, xx)
            xj = np.interp(j, tt, xx)
            yi = np.interp(i, tt, yy)
            yj = np.interp(j, tt, yy)

            dist = px_to_mm * np.sqrt((xj - xi)**2 + (yj - yi)**2)
            dists.append(dist)
            bend_i += 1
            j = i

    if len(dists) > 0:
        return np.sum(dists) / len(dists)
    else:
        return np.nan


def write_stats(settings, results, f, dead_stats=True, prepend='', mask=None):
    stats = statistics(results, settings, mask)
    s = stats

    if dead_stats:
        f.write('\n CUT-OFF tool/filters')
        f.write('\n-------------------------------\n')
        f.write('Total particles: %i\n' % s['original_particles'])
        f.write('Max particles present at same time: %i\n'
                % s['max_number_worms_present'])
        f.write('\n')

        if settings["cutoff_filter"]:
            f.write('Frame number:       ')

            for item in results["frames"]:
                f.write('%i,    ' % item)

            f.write('\n# of particles:   ')

            for item in results["list_number"]:
                f.write('%i,    ' % item)

            f.write('\nCut-off tool: Yes\n')
            if settings["use_average"]:
                f.write('Method: averaging\n')
            else:
                f.write('Method: maximum\n')
            f.write('Removed particles: %i\n' % s['removed_particles_cutoff'])
        else:
            f.write('Cut-off tool: No\n')

        if settings["extra_filter"]:
            f.write('Extra filter: Yes\n')
            f.write(
                'Settings: remove when bpm > %.5f and velocity < %.5f\n' % (
                    settings["Bends_max"], settings["Speed_max"]))
            f.write('Removed particles: %i' % s['spurious_worms'])
        else:
            f.write('Extra filter: No\n')
        f.write(prepend + '\n-------------------------------\n\n')

    f.write(prepend + 'BPM Mean: %.5f\n' % s['bpm_mean'])
    f.write(prepend + 'BPM Standard deviation: %.5f\n' % s['bpm_std'])
    f.write(prepend + 'BPM Error on mean: %.5f\n' % s['bpm_mean_std'])
    f.write(prepend + 'BPM Median: %.5f\n' % s['bpm_median'])

    f.write(prepend + 'Bends in movie Mean: %.5f\n' % s['bendsinmovie_mean'])
    f.write(prepend + 'Bends in movie Standard deviation: %.5f\n' %
            s['bendsinmovie_std'])
    f.write(prepend + 'Bends in movie Error on mean: %.5f\n' %
            s['bendsinmovie_mean_std'])
    f.write(
        prepend +
        'Bends in movie Median: %.5f\n' %
        s['bendsinmovie_median'])

    f.write(prepend + 'Speed Mean: %.6f\n' % s['vel_mean'])
    f.write(prepend + 'Speed Standard deviation: %.6f\n' % s['vel_std'])
    f.write(prepend + 'Speed Error on mean: %.6f\n' % s['vel_mean_std'])
    f.write(prepend + 'Speed Median: %.6f\n' % s['vel_median'])

    f.write(
        prepend +
        '90th Percentile speed mean: %.6f\n' %
        s['max_speed_mean'])
    f.write(prepend + '90th Percentile speed SD: %.6f\n' % s['max_speed_std'])
    f.write(prepend + '90th Percentile speed SEM: %.6f\n' %
            s['max_speed_mean_std'])
    if np.isnan(s['move_per_bend_mean']):
        f.write(prepend + 'Dist per bend Mean: nan\n')
        f.write(prepend + 'Dist per bend Standard deviation: nan\n')
        f.write(prepend + 'Dist per bend Error on mean: nan\n')
    else:
        f.write(
            prepend +
            'Dist per bend Mean: %.6f\n' %
            s['move_per_bend_mean'])
        f.write(prepend + 'Dist per bend Standard deviation: %.6f\n' %
                s['move_per_bend_std'])
        f.write(prepend + 'Dist per bend Error on mean: %.6f\n' %
                s['move_per_bend_mean_std'])
    if dead_stats:
        f.write(prepend + 'Live worms: %i\n' % s['n_live'])
        f.write(prepend + 'Dead worms: %i\n' % s['n_dead'])
        f.write(prepend + 'Total worms: %i\n' % s['max_number_worms_present'])
        f.write(prepend + 'Live ratio: %.6f\n' %
                (float(s['n_live']) / s['count']))
        f.write(prepend + 'Dead ratio: %.6f\n' %
                (float(s['n_dead']) / s['count']))
        if s['n_dead'] > 0:
            f.write(prepend + 'Live-to-dead ratio: %.6f\n' % (float(
                s['n_live']) / s['n_dead']))
        else:
            f.write(prepend + 'Live-to-dead ratio: inf\n')
        if s['n_live'] > 0:
            f.write(prepend + 'Dead-to-live ratio: %.6f\n' % (float(
                s['n_dead']) / s['n_live']))
        else:
            f.write(prepend + 'Dead-to-live ratio: inf\n')
    f.write(prepend + 'Area Mean: %.6f\n' % s['area_mean'])
    f.write(prepend + 'Area Standard Deviation: %.6f\n' % s['area_std'])
    f.write(prepend + 'Area Error on Mean: %.6f\n' % s['area_mean_std'])

    f.write(prepend + 'Round ratio mean: %.6f\n' % s['round_ratio_mean'])
    f.write(prepend + 'Round ratio SD: %.6f\n' % s['round_ratio_std'])
    f.write(prepend + 'Round ratio SEM: %.6f\n' % s['round_ratio_mean_std'])

    f.write(prepend + 'Eccentricity mean: %.6f\n' % s['eccentricity_mean'])
    f.write(prepend + 'Eccentricity SD: %.6f\n' % s['eccentricity_std'])
    f.write(prepend + 'Eccentricity SEM: %.6f\n' % s['eccentricity_mean_std'])


def write_raw_data(f, results, settings):
    bpm = results["bpm"]
    velocites = results["velocites"]
    move_per_bends = results["move_per_bends"]
    living = np.logical_or(
        bpm > settings["maximum_bpm"],
        velocites > settings["maximum_velocity"])
    x = (settings["limit_images_to"] / settings["fps"])

    f.write('Raw data:\n')
    f.write('Particle;BPM;Bends per %.2f s;Speed;Max speed;Dist per bend;'
            'Area;Appears in frames;Living;Region;Round ratio;'
            'Eccentricity \n' % x)
    f.write('\n'
            .join(['%i;%.6f;%.6f;%.6f;%.6f;%s;%.6f;%i;%i;%s;%.6f;%.6f'
                   % (
        i,
        results["bpm"][i],
        results["bendsinmovie"][i],
        results["velocites"][i],
        results["max_speed"][i],
        ('nan' if np.isnan(move_per_bends[i]) else '%.6f' % move_per_bends[i]),
        results["areas"][i],
        results["appears_in"][i],
        living[i],
        results["region"][i],
        results["round_ratio"][i],
        results["eccentricity"][i])
                   for i in range(len(bpm))]))
    f.write('\n\n')


def mean_std(x, appears_in):
    mean = np.sum(x * appears_in) / np.sum(appears_in)
    second_moment = np.sum(x**2 * appears_in) / np.sum(appears_in)
    std = np.sqrt(second_moment - mean**2)
    return mean, std


def statistics(results, settings, mask=None):
    if mask is None:
        mask = np.ones(len(results['bends'])) > 0

    particles = results["particles"][mask]
    appears_in = results["appears_in"][mask]
    bendsinmovie = results["bendsinmovie"][mask]
    areas = results["areas"][mask]
    max_speed = results["max_speed"][mask]
    round_ratio = results["round_ratio"][mask]
    eccentricity = results["eccentricity"][mask]
    move_per_bends = results["move_per_bends"][mask]
    spurious_worms = results["spurious_worms"]
    frames = results["frames"]
    list_number = results["list_number"]
    original_particles = results["original_particles"]
    P = track['particle']
    T = track['frame']

    if settings["cutoff_filter"]:
        max_number_worms_present = len(particles)
    else:
        max_number_worms_present = max(
            [len([1 for p in set(P[T == t]) if p in particles])
             for t in set(T)])
    count = len(particles)
    bpm = results["bpm"]
    velocites = results["velocites"]
    n_dead = np.sum(np.logical_and(
        bpm <= settings["maximum_bpm"],
        velocites <= settings["maximum_velocity"]))
    n_live = len(particles) - n_dead

    removed_particles_cutoff = original_particles - len(particles)

    bpm_mean, bpm_std = mean_std(bpm, appears_in)
    bpm_median = np.median(bpm)
    bpm_mean_std = bpm_std / np.sqrt(max_number_worms_present)

    bendsinmovie_mean, bendsinmovie_std = mean_std(bendsinmovie, appears_in)
    bendsinmovie_median = np.median(bendsinmovie)
    bendsinmovie_mean_std = bendsinmovie_std / \
        np.sqrt(max_number_worms_present)

    vel_mean, vel_std = mean_std(velocites, appears_in)
    vel_mean_std = vel_std / np.sqrt(max_number_worms_present)
    vel_median = np.median(velocites)

    area_mean, area_std = mean_std(areas, appears_in)
    area_mean_std = area_std / np.sqrt(max_number_worms_present)

    max_speed_mean, max_speed_std = mean_std(max_speed, appears_in)
    max_speed_mean_std = max_speed_std / np.sqrt(max_number_worms_present)

    round_ratio_mean, round_ratio_std = mean_std(round_ratio, appears_in)
    round_ratio_mean_std = round_ratio_std / np.sqrt(max_number_worms_present)

    eccentricity_mean, eccentricity_std = mean_std(eccentricity, appears_in)
    eccentricity_mean_std = eccentricity_std / \
        np.sqrt(max_number_worms_present)

    # Ignore nan particles for move_per_bend
    move_appear = [(move_per_bends[i], appears_in[i]) for i in range(len(
        appears_in)) if not np.isnan(move_per_bends[i])]
    if len(move_appear) > 0:
        mo, ap = list(zip(*move_appear))
        move_per_bend_mean, move_per_bend_std = mean_std(np.array(mo),
                                                         np.array(ap))
        move_per_bend_mean_std = move_per_bend_std / \
            np.sqrt(max([len(mo), max_number_worms_present]))
    else:
        move_per_bend_mean = np.nan
        move_per_bend_std = np.nan
        move_per_bend_mean_std = np.nan

    stats = { 'max_number_worms_present': max_number_worms_present,
              'n_dead': n_dead,
              'n_live': n_live,
              'bpm_mean': bpm_mean,
              'bpm_std': bpm_std,
              'bpm_std': bpm_std,
              'bpm_median': bpm_median,
              'bpm_mean_std': bpm_mean_std,
              'bendsinmovie_mean': bendsinmovie_mean,
              'bendsinmovie_std': bendsinmovie_std,
              'bendsinmovie_mean_std': bendsinmovie_mean_std,
              'bendsinmovie_median': bendsinmovie_median,
              'vel_mean': vel_mean,
              'vel_std': vel_std,
              'vel_mean_std': vel_mean_std,
              'vel_median': vel_median,
              'area_mean': area_mean,
              'area_std': area_std,
              'area_mean_std': area_mean_std,
              'max_speed_mean': max_speed_mean,
              'max_speed_std': max_speed_std,
              'max_speed_mean_std': max_speed_mean_std,
              'move_per_bend_mean': move_per_bend_mean,
              'move_per_bend_std': move_per_bend_std,
              'move_per_bend_mean_std': move_per_bend_mean_std,
              'removed_particles_cutoff': removed_particles_cutoff,
              'spurious_worms': spurious_worms,
              'frames': frames,
              'list_number': list_number,
              'original_particles': original_particles,
              'count': count,
              'round_ratio_mean': round_ratio_mean,
              'round_ratio_std': round_ratio_std,
              'round_ratio_mean_std': round_ratio_mean_std,
              'eccentricity_mean': eccentricity_mean,
              'eccentricity_std': eccentricity_std,
              'eccentricity_mean_std': eccentricity_mean_std}

    return stats


def write_results_file(results, settings):
    '''
    Input:
        region_particles: list of particles contained in a region
        track: the full track, used to calculate
                        maximum number of worms present
                        at any given time, which is ised in statistics
                         to avoid underestimation of errors.

        The remaining input parameters all have the same shape
        corresponding to different particles.
    '''

    with open(os.path.join(settings["save_as"], 'results.txt'), 'w') as f:
        f.write('---------------------------------\n')
        f.write('    Results for %s \n' % settings["video_filename"])
        f.write('---------------------------------\n\n')

        # Stats for all worms
        write_stats(settings, results, f, dead_stats=True)

        # Stats for living worms
        living_mask = np.logical_or(
            results["bpm"] > settings["maximum_bpm"],
            results["velocites"] > settings["maximum_velocity"])


        write_stats(settings, results, f, dead_stats=False, prepend='Living ',
                    mask=living_mask)

        # Raw stats
        f.write('---------------------------------\n\n')
        write_raw_data(f, results, settings)

        regions = settings["regions"]
        # Per region stats
        if len(regions) > 1:
            for reg in regions:
                f.write('---------------------------------\n')
                f.write('Stats for region: %s\n' % reg)
                f.write('---------------------------------\n\n')

                # Worms of this region
                try:
                    pars = list(map(int, results["region_particles"][reg]))
                except TypeError:
                    pars = [int(results["region_particles"][reg])]
                if len(pars) == 0:
                    f.write('Nothing found in region.\n\n')
                    continue
                indeces = [i for i, p in enumerate(particles) if p in pars]
                region_mask = np.ones_like(results["areas"]) == 0
                region_mask[indeces] = 1

                # All worms
                write_stats(settings, results, f, dead_stats=True,
                            mask=region_mask)

                f.write('\n\n')
        f.write('\n')

    print('results.txt file produced.')

# =============================================================================
# --- Matplotlib code---
# =============================================================================
def print_frame(settings, t, particles, P, T, bends, track):
    font = {'size': settings["font_size"]}
    print('Printing frame', t + 1)
    frame = (255 - io.imread(
        os.path.join(settings["save_as"],
        'imgs','%05d.jpg' % (int(t)))))
    small_imshow(frame, cmap=cm.binary, vmax=300)
    for i, p in enumerate(particles):
        pp = P == p
        l = np.logical_and(pp, T == t)
        if np.sum(l) > 0:
            x = track['x'][l].iloc[0]
            y = track['y'][l].iloc[0]
            b = bends[i][np.sum(T[pp] < t)]
            plt.text(y + 3, x + 3, 'p=%i\n%.1f' %
                     (i, b), font, color=[1, 0.3, 0.2])

    m, n = frame.shape
    plt.plot(
        [n - (5 + settings["scale_bar_size"] / float(settings["px_to_mm"])),
         n - 5],
        [m - 5, m - 5],
        linewidth=settings["scale_bar_thickness"], c=[0.5, 0.5, 0.5])
    plt.axis('off')
    plt.axis('tight')
    plt.savefig(os.path.join(settings["save_as"], 'imgs','%05d.jpg' % (t)))


def print_images(settings, results):
    track = results["track"]
    plt.figure(figsize=settings["fig_size"])
    P = track['particle']
    T = track['frame']
    output_overlayed_images = settings["output_overlayed_images"]
    if output_overlayed_images != 0:
        up_to = (len(set(T)) if output_overlayed_images is None
                 else output_overlayed_images)
        for t in range(up_to):
            print_frame(t, results["particles"], P, T, results["bends"], track)


def small_imshow(img, *args, **kwargs):
    # For large images/frames matplotlib's imshow gives memoryerror
    # This is solved by resizing before plotting
    s = img.shape
    b = img
    if (s[0] + s[1]) / 2. > 1500:
        factor = 1500 / ((s[0] + s[1]) / 2.)
        b = resize(img, (int(s[0] * factor), int(s[1] * factor)),
                   preserve_range=True)
    plt.clf()
    plt.imshow(b, *args, extent=[0, s[1], s[0], 0], **kwargs)


def output_processing_frames(
        settings, save_folder, frameorig, Z, frame, thresholded,
        frame_after_open, frame_after_close, labeled,
        labeled_removed, skel_labeled=None):
    plt.figure(figsize=settings["fig_size"])
    small_imshow(frameorig, cmap=cm.gray)
    plt.savefig(os.path.join(save_folder, '0frameorig.jpg'))

    small_imshow(Z, cmap=cm.gray)
    plt.savefig(os.path.join(save_folder, '0z.jpg'))

    small_imshow(frame, cmap=cm.gray)
    plt.savefig(os.path.join(save_folder, '1framesubtract.jpg'))

    small_imshow(thresholded, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '2thresholded.jpg'))

    small_imshow(frame_after_open, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '3opened.jpg'))

    small_imshow(frame_after_close, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '4closed.jpg'))

    small_imshow(labeled, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '5labelled.jpg'))

    small_imshow(labeled_removed, cmap=cm.binary)
    plt.savefig(os.path.join(save_folder, '6removed.jpg'))

    if skel_labeled is not None:
        small_imshow(skel_labeled, cmap=cm.binary)
        plt.savefig(os.path.join(save_folder, '7skeletonized.jpg'))
    plt.clf()

def print_example_frame(
        settings, sizes, save_folder, frameorig, Z, frame, thresholded,
        frame_after_open, frame_after_close, labeled, labeled_removed,
        skel_labeled):
    print('Sizes:')
    print(sizes)

    output_processing_frames(
        settings, save_folder, frameorig, Z, frame, thresholded,
        frame_after_open, frame_after_close, labeled, labeled_removed,
        (skel_labeled if settings["skeletonize"] else None))
    print('Example frame outputted!')
from pathlib import Path
import spatialdata as sd
import pandas as pd
import tifffile
import numpy as np
import dask.dataframe as dd
import xarray as xr
from scipy.linalg import lstsq
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from skimage.measure import regionprops_table
import anndata as ad
import dask
import zarr
import time
from spatialdata._io._utils import _resolve_zarr_store
from tqdm.notebook import tqdm as tqdm_nb

from spatialdata.transformations import (
    get_transformation,
    set_transformation,
    Identity,
    Sequence,
    Scale,
    Affine,
)

from spatialdata.models import (
    PointsModel,
    TableModel,
    Labels3DModel,
    Image3DModel,
)

from xenium_analysis_tools.utils.sd_utils import (
    add_micron_coord_sys,
)

from xenium_analysis_tools.alignment.align_sections import (
    get_affine_from_landmarks_flat,
    tilt_affines
)

def get_section_z_stats(affines, landmarks, czstack_shape_yx=(512, 512),
                        czstack_xy_um=1.0, czstack_z_um_per_plane=1.0):
    """
    Parameters
    ----------
    czstack_xy_um : float
        Physical size of one czstack XY pixel in µm.  Used to convert the tilt
        coefficients (czstack_z_planes / section_px) to a true geometric angle.
    czstack_z_um_per_plane : float
        Physical thickness of one czstack z-plane in µm.
    """
    y_max, x_max = czstack_shape_yx

    # FOV corners in czstack (x, y) pixels
    cz_corners = np.array([
        [0,     0    ],
        [x_max, 0    ],
        [0,     y_max],
        [x_max, y_max],
    ], dtype=float)

    results = {}
    for s_n, section_affines in affines.items():
        section_landmarks = landmarks[f'landmarks-{s_n}']
        tilt = section_affines['section_to_czstack_full_affine']
        czstack_pts = section_landmarks[['czstack_x', 'czstack_y', 'czstack_z']].values
        if isinstance(tilt, dict):
            tilt = Affine(
                matrix=np.array(tilt['matrix']),
                input_axes=tuple(tilt['input_axes']),
                output_axes=tuple(tilt['output_axes'])
            )
        mat = tilt.matrix  # (4,4), input_axes=(x, y, z)

        # ── Tilt coefficients: z_czstack = a*x_xe + b*y_xe + c ───────────────
        # mat[0,2] = mat[1,2] = 0 by construction, so XY mapping is z-independent.
        a_z = mat[2, 0]   # czstack_z_planes / section_x_px
        b_z = mat[2, 1]   # czstack_z_planes / section_y_px
        c_z = mat[2, 3]   # z at section origin

        # ── Invert only the 2×2 XY submatrix: czstack_px → section_px ────────
        xy_sub = mat[:2, :2]
        t_xy   = mat[:2, 3]
        inv_xy = np.linalg.inv(xy_sub)

        # Section (x, y) at each czstack FOV corner
        xe_corners = (cz_corners - t_xy) @ inv_xy.T   # (4, 2)

        # Z (czstack planes) at each FOV corner
        z_corners = a_z * xe_corners[:, 0] + b_z * xe_corners[:, 1] + c_z

        # Z at FOV centre
        cz_center = np.array([[x_max / 2, y_max / 2]])
        xe_center = (cz_center - t_xy) @ inv_xy.T     # (1, 2)
        z_at_center = float(a_z * xe_center[0, 0] + b_z * xe_center[0, 1] + c_z)

        # Landmark-derived z centroid
        z_centroid_from_landmarks = float(np.mean(czstack_pts[:, 2]))

        # ── True geometric tilt angle ─────────────────────────────────────────
        # (a_z, b_z) are in czstack_z_planes / section_px — mixed units.
        # Physical lateral gradient (µm of z-rise per µm of lateral travel):
        #   section_px → czstack_px : scale_xy  (column-0 norm of XY submatrix)
        #   czstack_z_planes → µm   : czstack_z_um_per_plane
        #   czstack_px → µm         : czstack_xy_um
        scale_xy = float(np.linalg.norm(mat[:2, 0]))   # czstack_px / section_px
        tilt_gradient = (
            np.sqrt(a_z ** 2 + b_z ** 2)
            * czstack_z_um_per_plane
            * scale_xy
            / czstack_xy_um
        )
        tilt_deg = float(np.degrees(np.arctan(tilt_gradient)))

        results[s_n] = {
            'z_centroid_landmarks': z_centroid_from_landmarks,
            'z_at_fov_center':      z_at_center,
            'z_min_in_fov':         float(z_corners.min()),
            'z_max_in_fov':         float(z_corners.max()),
            'z_span_in_fov':        float(z_corners.max() - z_corners.min()),
            'tilt_deg':             tilt_deg,
        }
    return results

def print_z_stats_and_check_overlaps(results, 
                                      czstack_pixel_um=0.78125,
                                      czstack_z_um_per_plane=1.0,
                                      sections_depth_um=20.0,
                                      z_units='planes'):
    """
    Parameters
    ----------
    results : dict from get_section_z_stats
        z values are in czstack planes (z_um_per_plane = 1.0 for this dataset)
    czstack_pixel_um : float
        xy pixel size in µm (default 0.78125)
    czstack_z_um_per_plane : float
        z step size in µm per plane (default 1.0)
    sections_depth_um : float
        expected physical section thickness in µm
    z_units : str
        'planes' or 'microns' — controls display units for z values.
        For this dataset they are equivalent (1 plane = 1 µm) but
        labeling them correctly avoids ambiguity.
    """
    assert z_units in ('planes', 'microns'), "z_units must be 'planes' or 'microns'"
    zu = 'µm' if z_units == 'microns' else 'pl'

    # Convert planes → µm if requested
    scale = czstack_z_um_per_plane if z_units == 'microns' else 1.0

    print(f"{'Sec':>4} {'z_lm_centroid':>16} {'z_fov_center':>14} "
          f"{'z_min':>9} {'z_max':>9} {'z_span':>9} {'tilt°':>6}")
    print(f"{'':>4} {'('+zu+')':>16} {'('+zu+')':>14} "
          f"{'('+zu+')':>9} {'('+zu+')':>9} {'('+zu+')':>9}")
    print("-" * 68)
    for s_n, r in sorted(results.items()):
        print(f"{s_n:>4} "
              f"{r['z_centroid_landmarks']*scale:>16.1f} "
              f"{r['z_at_fov_center']*scale:>14.1f} "
              f"{r['z_min_in_fov']*scale:>9.1f} "
              f"{r['z_max_in_fov']*scale:>9.1f} "
              f"{r['z_span_in_fov']*scale:>9.1f} "
              f"{r['tilt_deg']:>6.2f}°")

    print(f"\nOverlap check (consecutive sections):")
    print(f"Note: z values are in czstack {z_units} "
          f"({czstack_z_um_per_plane} µm/plane, "
          f"xy pixel = {czstack_pixel_um} µm)")

    sorted_sections = sorted(results.keys())
    for i in range(len(sorted_sections) - 1):
        s_a = sorted_sections[i]
        s_b = sorted_sections[i + 1]

        # Separation in planes (raw from affine), then convert to µm
        sep_planes = abs(results[s_b]['z_at_fov_center'] - 
                         results[s_a]['z_at_fov_center'])
        sep_um     = sep_planes * czstack_z_um_per_plane
        min_sep_um = sections_depth_um

        # Overlap test in planes (raw units, no conversion needed)
        fov_overlap = results[s_a]['z_max_in_fov'] > results[s_b]['z_min_in_fov']

        print(f"  Sections {s_a}→{s_b}: "
              f"separation = {sep_planes:.1f} pl "
              f"= {sep_um:.1f} µm  "
              f"(min expected {min_sep_um:.1f} µm)  "
              f"{'⚠  FOV ranges overlap' if fov_overlap else '✓  non-overlapping'}")

        results[s_a]['overlap_with_next']       = fov_overlap
        results[s_a]['separation_planes_to_next'] = sep_planes
        results[s_a]['separation_um_to_next']     = sep_um

    results[sorted_sections[-1]]['overlap_with_next']         = False
    results[sorted_sections[-1]]['separation_planes_to_next'] = None
    results[sorted_sections[-1]]['separation_um_to_next']     = None

    return results

def find_landmark_outliers(landmarks, czstack_xy_um=1.0, czstack_z_um_per_plane=1.0):
      if isinstance(landmarks, dd.DataFrame):
            lm = landmarks.compute()
      else:
            lm = landmarks
      
      cz_z = lm['czstack_z'].values

      # Identify outliers: landmarks more than 2 std from mean
      mean_z, std_z = cz_z.mean(), cz_z.std()
      outlier_mask = np.abs(cz_z - mean_z) > 2 * std_z
      print(f"Outlier landmarks ({outlier_mask.sum()} found):")
      print(lm[outlier_mask][['czstack_x', 'czstack_y', 'czstack_z', 'x', 'y']])

      # Compare affine with and without outliers
      xenium_pts_all = lm[['x', 'y', 'z']].values
      czstack_pts_all = lm[['czstack_x', 'czstack_y', 'czstack_z']].values

      xenium_pts_clean = xenium_pts_all[~outlier_mask]
      czstack_pts_clean = czstack_pts_all[~outlier_mask]

      flat_all   = get_affine_from_landmarks_flat(xenium_pts_all, czstack_pts_all)
      flat_clean = get_affine_from_landmarks_flat(xenium_pts_clean, czstack_pts_clean)
      tilt_all   = tilt_affines(xenium_pts_all, czstack_pts_all, flat_all)
      tilt_clean = tilt_affines(xenium_pts_clean, czstack_pts_clean, flat_clean)

      def _tilt_deg(tilt_mat, flat_mat):
          """True geometric tilt angle in degrees, accounting for physical pixel scales."""
          a_z, b_z   = tilt_mat.matrix[2, 0], tilt_mat.matrix[2, 1]
          scale_xy   = float(np.linalg.norm(flat_mat.matrix[:2, 0]))  # czstack_px / section_px
          grad = np.sqrt(a_z**2 + b_z**2) * czstack_z_um_per_plane * scale_xy / czstack_xy_um
          return np.degrees(np.arctan(grad))

      print(f"\nWith all landmarks:     z_offset = {flat_all.matrix[2,3]:.3f}, "
            f"tilt = {_tilt_deg(tilt_all, flat_all):.3f}°")
      print(f"Without outliers:       z_offset = {flat_clean.matrix[2,3]:.3f}, "
            f"tilt = {_tilt_deg(tilt_clean, flat_clean):.3f}°")


def diagnose_section_z_placement(results, sections_um=20.0, czstack_z_um_per_plane=1.0,
                                  residual_threshold_um=5.0):
    """
    Identify sections likely mis-placed in z by fitting a robust linear trend
    through z_at_fov_center vs section number and reporting residuals.

    Uses Theil-Sen regression (median of all pairwise slopes), which is robust
    to ~50% outliers — so close-pair placement errors don't bias the estimated
    true section spacing.

    Parameters
    ----------
    results : dict
        Output of get_section_z_stats (must have run print_z_stats_and_check_overlaps
        first so separation fields are populated).
    sections_um : float
        Expected physical section thickness in µm.
    czstack_z_um_per_plane : float
        µm per czstack z-plane.
    residual_threshold_um : float
        Sections with |residual| > this value are flagged for z correction.

    Returns
    -------
    corrections : dict {section_n: delta_z_planes}
        Add delta to every czstack_z landmark of that section to shift it onto
        the fitted line.  Positive delta = section was placed too shallow.
    """
    from scipy.stats import theilslopes

    sections = sorted(results.keys())
    z_vals = np.array([results[s]['z_at_fov_center'] for s in sections])
    s_arr  = np.array(sections, dtype=float)

    # Robust linear fit — resistant to the close-pair errors
    res       = theilslopes(z_vals, s_arr)
    slope, intercept = res.slope, res.intercept
    z_fitted  = slope * s_arr + intercept
    residuals = z_vals - z_fitted          # positive = section placed too deep

    expected_slope = sections_um / czstack_z_um_per_plane
    threshold_pl   = residual_threshold_um / czstack_z_um_per_plane

    print(f"Robust linear fit (Theil-Sen):  z = {slope:.2f} × section + {intercept:.2f}")
    print(f"  fitted spacing:   {slope:.2f} pl/section = {slope * czstack_z_um_per_plane:.1f} µm/section")
    print(f"  expected spacing: {expected_slope:.2f} pl/section = {sections_um:.1f} µm/section\n")

    print(f"{'Sec':>4}  {'z_actual':>9}  {'z_fitted':>9}  {'resid_pl':>9}  {'resid_µm':>9}  status")
    print("─" * 66)

    corrections = {}
    for s, z_a, z_f, r in zip(sections, z_vals, z_fitted, residuals):
        r_um = r * czstack_z_um_per_plane
        flag = "⚠  ADJUST" if abs(r_um) > residual_threshold_um else "✓"
        print(f"{s:>4}  {z_a:>9.1f}  {z_f:>9.1f}  {r:>+9.1f}  {r_um:>+8.1f}µm  {flag}")
        if abs(r_um) > residual_threshold_um:
            corrections[s] = float(-r)   # delta to add to czstack_z to reach fitted position

    print(f"\n{len(corrections)} section(s) flagged  (|residual| > {residual_threshold_um} µm)")
    if corrections:
        print("Suggested z corrections (czstack planes to add to every czstack_z landmark):")
        for s, delta in sorted(corrections.items()):
            corrected_z = results[s]['z_at_fov_center'] + delta
            print(f"  Section {s:>3}:  {delta:+.1f} pl  ({delta * czstack_z_um_per_plane:+.1f} µm)"
                  f"  →  corrected z_center = {corrected_z:.1f}")

    return corrections


def apply_section_z_corrections(sections_landmarks, sections_affines, corrections,
                                 czstack_z_um_per_plane=1.0):
    """
    Apply z-only shifts to landmarks and recompute affines for flagged sections.

    Only czstack_z landmark coordinates are modified — x/y are untouched, and the
    XY components of every affine are unchanged.  The tilt coefficients (mat[2,0],
    mat[2,1]) are preserved; only the z-offset constant (mat[2,3]) shifts because
    get_alignment_transforms re-fits it from the corrected czstack_z values.

    Parameters
    ----------
    sections_landmarks : sd.SpatialData
        Points-only SpatialData with keys 'landmarks-{s_n}'.
    sections_affines : dict {section_n: affines_dict}
        Original affines from get_alignment_transforms.
    corrections : dict {section_n: delta_z_planes}
        Output of diagnose_section_z_placement, or manually specified.
    czstack_z_um_per_plane : float
        Used only for the printed summary.

    Returns
    -------
    corrected_lm_dict : dict {section_n: pd.DataFrame}
        All sections, corrected ones with shifted czstack_z.
    corrected_affines : dict {section_n: affines_dict}
        Recomputed affines for corrected sections; original affines for the rest.
    """
    from xenium_analysis_tools.alignment.align_sections import get_alignment_transforms

    corrected_lm_dict  = {}
    corrected_affines  = {}

    print("Applying z corrections:")
    for s_n, aff in sorted(sections_affines.items()):
        lm = sections_landmarks.points[f'landmarks-{s_n}']
        lm = lm.compute() if hasattr(lm, 'compute') else lm.copy()

        if s_n in corrections:
            delta = corrections[s_n]
            lm = lm.copy()
            lm['czstack_z'] = lm['czstack_z'] + delta
            corrected_affines[s_n] = get_alignment_transforms(lm)
            print(f"  Section {s_n:>3}: czstack_z shifted {delta:+.2f} pl "
                  f"({delta * czstack_z_um_per_plane:+.2f} µm)  → affines recomputed")
        else:
            corrected_affines[s_n] = aff

        corrected_lm_dict[s_n] = lm

    print(f"\nDone. {len(corrections)} section(s) corrected, "
          f"{len(sections_affines) - len(corrections)} unchanged.")
    return corrected_lm_dict, corrected_affines


def _build_section_colors(sections, pairs):
    """Return a {section_n: hex_color} dict, shared by both plot functions."""
    pair_hues = [
        ('#4e9af1', '#1a5fa8'),
        ('#3db87a', '#1a6e46'),
        ('#e05c5c', '#8a1f1f'),
        ('#b07fd4', '#6a3a9a'),
        ('#50b8c4', '#1e6e7a'),
        ('#f4a742', '#9a5e10'),
    ]
    bright_palette = [
        '#4e9af1', '#3db87a', '#e05c5c', '#b07fd4',
        '#50b8c4', '#f4a742', '#f17c4e', '#a8d44e',
        '#f14eb0', '#4ef1d4', '#d4c44e', '#7a4ef1',
        '#f1d44e', '#4ef17a', '#f14e7a', '#4eaff1',
    ]
    colors = {}
    if pairs:
        pair_idx = 0
        for p in pairs:
            if pair_idx < len(pair_hues):
                colors[p[0]] = pair_hues[pair_idx][0]
                colors[p[1]] = pair_hues[pair_idx][1]
                pair_idx += 1
        bright_idx = pair_idx * 2
        for s_n in sections:
            if s_n not in colors:
                colors[s_n] = bright_palette[bright_idx % len(bright_palette)]
                bright_idx += 1
    else:
        for i, s_n in enumerate(sections):
            colors[s_n] = bright_palette[i % len(bright_palette)]
    return colors


def _draw_xz_side_view(ax, results, sections, pairs, colors, czstack_xy_um,
                        czstack_depth_um, czstack_x_shape, title):
    """Draw the XZ side-view panel onto *ax* (shared by plot functions)."""
    paired = {s for p in pairs for s in p}
    x_fov  = czstack_xy_um * czstack_x_shape

    ax.set_facecolor('#0f1117')
    ax.set_xlim(-5, x_fov + 100)
    ax.set_ylim(czstack_depth_um + 10, -10)
    ax.set_xlabel('x position in czstack (µm)', color='#8b9ab0', fontsize=9)
    ax.set_ylabel('z depth (µm)', color='#8b9ab0', fontsize=9)
    ax.set_title(title, color='#e2e8f0', fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')
    ax.tick_params(colors='#8b9ab0', labelsize=8)
    ax.yaxis.grid(True, color='#2d3748', linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)

    for s_a, s_b in pairs:
        if s_a not in results or s_b not in results:
            continue
        z_lo = min(results[s_a]['z_min_in_fov'], results[s_b]['z_min_in_fov'])
        z_hi = max(results[s_a]['z_max_in_fov'], results[s_b]['z_max_in_fov'])
        ax.axhspan(z_lo, z_hi, xmin=0, xmax=x_fov / (x_fov + 100),
                   color=colors[s_a], alpha=0.07, zorder=0)

    for s_n in sections:
        if s_n not in results:
            continue
        r   = results[s_n]
        col = colors[s_n]
        is_paired = s_n in paired
        ax.plot([0, x_fov], [r['z_min_in_fov'], r['z_max_in_fov']],
                color=col,
                linewidth=2.5 if is_paired else 1.2,
                solid_capstyle='round', alpha=0.95)
        ax.text(x_fov + 5, r['z_at_fov_center'],
                f'S{s_n}  {r["z_at_fov_center"]:.0f}µm  {r["tilt_deg"]:.2f}°',
                color=col, fontsize=6.5, va='center',
                fontweight='bold' if is_paired else 'normal')

    legend_handles = []
    for s_a, s_b in pairs:
        if s_a not in results:
            continue
        legend_handles.append(
            mlines.Line2D([], [], color=colors[s_a], linewidth=2.5,
                          label=f'S{s_a}+S{s_b} (paired)'))
    legend_handles.append(
        mlines.Line2D([], [], color='#9b9b9b', linewidth=1.2,
                      label='standalone sections'))
    ax.legend(handles=legend_handles, fontsize=7,
              facecolor='#1a2130', edgecolor='#2d3748',
              labelcolor='#8b9ab0', loc='lower left')


def plot_section_positions(results,
                           czstack_xy_um=0.78125,
                           czstack_depth_um=450,
                           czstack_x_shape=512,
                           pairs=None,
                           save_path=None):
    sections = sorted(results.keys())

    if pairs is None:
        pairs = []
    paired = {s for p in pairs for s in p}

    colors = _build_section_colors(sections, pairs)

    # ── Pair membership lookup ────────────────────────────────────────────
    pair_of = {}
    for p in pairs:
        pair_of[p[0]] = p
        pair_of[p[1]] = p

    fig = plt.figure(figsize=(15, 6.5), facecolor='#0f1117')
    gs = GridSpec(1, 2, figure=fig, wspace=0.44)

    # ── Left: side view (XZ) ─────────────────────────────────────────────
    ax_xz = fig.add_subplot(gs[0])
    _draw_xz_side_view(ax_xz, results, sections, pairs, colors,
                       czstack_xy_um, czstack_depth_um, czstack_x_shape,
                       title='Side view: section positions and tilt')

    # ── Right: z-position chart ───────────────────────────────────────────
    ax_z = fig.add_subplot(gs[1])
    ax_z.set_facecolor('#0f1117')
    for spine in ax_z.spines.values():
        spine.set_edgecolor('#2d3748')
    ax_z.tick_params(colors='#8b9ab0', labelsize=8)
    ax_z.xaxis.grid(True, color='#2d3748', linewidth=0.5, linestyle='--')
    ax_z.set_axisbelow(True)

    y_pos     = np.arange(len(sections))
    z_centers = [results[s]['z_at_fov_center'] for s in sections]
    z_mins    = [results[s]['z_min_in_fov']    for s in sections]
    z_spans   = [results[s]['z_span_in_fov']   for s in sections]
    bar_h = 0.5

    # Background band per pair (full x-width, subtle)
    for s_a, s_b in pairs:
        if s_a not in sections or s_b not in sections:
            continue
        i_a = sections.index(s_a)
        i_b = sections.index(s_b)
        ax_z.axhspan(i_a - bar_h, i_b + bar_h,
                     color=colors[s_a], alpha=0.07, zorder=0)

    # Bars and dots
    for i, s_n in enumerate(sections):
        col = colors[s_n]
        is_paired = s_n in paired
        ax_z.barh(i, z_spans[i], left=z_mins[i], height=bar_h,
                  color=col, alpha=0.5,
                  linewidth=1.5 if is_paired else 0.3,
                  edgecolor=col if is_paired else 'none')
        ax_z.scatter(z_centers[i], i, color=col,
                     s=50 if is_paired else 25, zorder=5,
                     marker='D' if is_paired else 'o')

    # Pair connectors: bracket between the two center dots
    for s_a, s_b in pairs:
        if s_a not in sections or s_b not in sections:
            continue
        i_a = sections.index(s_a)
        i_b = sections.index(s_b)
        # Use the shared hue (lighter of the two) for the bracket
        bracket_col = colors[s_a]
        x_bracket = z_mins[i_a] - 3  # just left of bars
        ax_z.plot([x_bracket, x_bracket], [i_a, i_b],
                  color=bracket_col, linewidth=1.5, alpha=0.6, zorder=2)
        ax_z.plot([x_bracket, x_bracket + 1.5], [i_a, i_a],
                  color=bracket_col, linewidth=1.5, alpha=0.6, zorder=2)
        ax_z.plot([x_bracket, x_bracket + 1.5], [i_b, i_b],
                  color=bracket_col, linewidth=1.5, alpha=0.6, zorder=2)

    # Amber overlap boxes (in data coords before invert_yaxis)
    overlap_labeled = False
    for i in range(len(sections) - 1):
        s_a, s_b = sections[i], sections[i + 1]
        z_max_a = results[s_a]['z_max_in_fov']
        z_min_b = results[s_b]['z_min_in_fov']
        if z_max_a > z_min_b:
            rect = plt.Rectangle(
                (z_min_b, i - bar_h / 2),
                z_max_a - z_min_b,
                (i + 1 + bar_h / 2) - (i - bar_h / 2),
                color='#f4a742', alpha=0.22, zorder=1,
                label='z-range overlap' if not overlap_labeled else None
            )
            ax_z.add_patch(rect)
            overlap_labeled = True

    ax_z.set_yticks(y_pos)
    ax_z.set_yticklabels([f'S{s}' for s in sections],
                          fontsize=7.5, color='#8b9ab0')
    ax_z.set_xlabel('z depth (µm)', color='#8b9ab0', fontsize=9)
    ax_z.set_title('Section z-positions  (bar = z-span, dot = center)',
                   color='#e2e8f0', fontsize=10, pad=8)

    z_all_min = min(z_mins) - 5
    z_all_max = max(r + s for r, s in zip(z_mins, z_spans)) + 5
    ax_z.set_xlim(z_all_min, z_all_max)
    ax_z.invert_yaxis()

    paired_bar    = mpatches.Patch(color='#8b9ab0', alpha=0.6,
                                    label='paired (edged bar, ◆)')
    solo_bar      = mpatches.Patch(color='#8b9ab0', alpha=0.25,
                                    label='standalone (plain bar, ●)')
    overlap_patch = mpatches.Patch(color='#f4a742', alpha=0.4,
                                    label='z-range overlap')
    ax_z.legend(handles=[paired_bar, solo_bar, overlap_patch],
                fontsize=7, facecolor='#1a2130',
                edgecolor='#2d3748', labelcolor='#8b9ab0',
                loc='lower right')

    fig.suptitle('Xenium section positions within cortical z-stack',
                 color='#e2e8f0', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    return fig                      


def plot_z_correction_comparison(results_original, results_corrected,
                                  czstack_xy_um=0.78125,
                                  czstack_depth_um=450,
                                  czstack_x_shape=512,
                                  pairs=None,
                                  corrections=None,
                                  save_path=None):
    """Side-by-side XZ side-view of section positions before and after z correction.

    Parameters
    ----------
    results_original : dict
        Output of `get_section_z_stats` before any correction.
    results_corrected : dict
        Output of `get_section_z_stats` after `apply_section_z_corrections`.
    czstack_xy_um, czstack_depth_um, czstack_x_shape : float / int
        Same physical parameters as `plot_section_positions`.
    pairs : list of (int, int), optional
        Paired section indices (same convention as `plot_section_positions`).
    corrections : dict {section_n: delta_z_planes}, optional
        Sections that were moved.  They are marked with a star ('*') in the
        corrected panel title and with a dashed line style.
    save_path : str or Path, optional
        If provided the figure is saved there at 180 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sections = sorted(set(results_original) | set(results_corrected))
    if pairs is None:
        pairs = []
    if corrections is None:
        corrections = {}

    # Shared color scheme so corresponding sections look the same on both panels
    colors = _build_section_colors(sections, pairs)

    fig = plt.figure(figsize=(14, 6.5), facecolor='#0f1117')
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Left: original ────────────────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0])
    _draw_xz_side_view(ax_orig, results_original, sections, pairs, colors,
                       czstack_xy_um, czstack_depth_um, czstack_x_shape,
                       title='Original z-placement')

    # ── Right: corrected ──────────────────────────────────────────────────
    ax_corr = fig.add_subplot(gs[1])
    ax_corr.set_facecolor('#0f1117')
    for spine in ax_corr.spines.values():
        spine.set_edgecolor('#2d3748')
    ax_corr.tick_params(colors='#8b9ab0', labelsize=8)
    ax_corr.yaxis.grid(True, color='#2d3748', linewidth=0.5, linestyle='--')
    ax_corr.set_axisbelow(True)

    paired = {s for p in pairs for s in p}
    x_fov  = czstack_xy_um * czstack_x_shape

    ax_corr.set_xlim(-5, x_fov + 100)
    ax_corr.set_ylim(czstack_depth_um + 10, -10)
    ax_corr.set_xlabel('x position in czstack (µm)', color='#8b9ab0', fontsize=9)
    ax_corr.set_ylabel('z depth (µm)', color='#8b9ab0', fontsize=9)
    n_corr = len(corrections)
    corr_subtitle = (f'Corrected ({n_corr} section{"s" if n_corr != 1 else ""} adjusted)'
                     if n_corr else 'Corrected (no changes)')
    ax_corr.set_title(corr_subtitle, color='#e2e8f0', fontsize=10, pad=8)

    # Shade paired z-bands from the corrected positions
    for s_a, s_b in pairs:
        if s_a not in results_corrected or s_b not in results_corrected:
            continue
        z_lo = min(results_corrected[s_a]['z_min_in_fov'],
                   results_corrected[s_b]['z_min_in_fov'])
        z_hi = max(results_corrected[s_a]['z_max_in_fov'],
                   results_corrected[s_b]['z_max_in_fov'])
        ax_corr.axhspan(z_lo, z_hi, xmin=0, xmax=x_fov / (x_fov + 100),
                        color=colors[s_a], alpha=0.07, zorder=0)

    for s_n in sections:
        if s_n not in results_corrected:
            continue
        r   = results_corrected[s_n]
        col = colors[s_n]
        is_paired    = s_n in paired
        was_corrected = s_n in corrections
        ls  = '--' if was_corrected else '-'
        lw  = 2.5 if is_paired else 1.2
        ax_corr.plot([0, x_fov], [r['z_min_in_fov'], r['z_max_in_fov']],
                     color=col, linewidth=lw, linestyle=ls,
                     solid_capstyle='round', alpha=0.95)
        star = ' ★' if was_corrected else ''
        ax_corr.text(x_fov + 5, r['z_at_fov_center'],
                     f'S{s_n}{star}  {r["z_at_fov_center"]:.0f}µm  {r["tilt_deg"]:.2f}°',
                     color=col, fontsize=6.5, va='center',
                     fontweight='bold' if is_paired else 'normal')

    # Legend
    legend_handles = []
    for s_a, s_b in pairs:
        if s_a not in results_corrected:
            continue
        legend_handles.append(
            mlines.Line2D([], [], color=colors[s_a], linewidth=2.5,
                          label=f'S{s_a}+S{s_b} (paired)'))
    legend_handles.append(
        mlines.Line2D([], [], color='#9b9b9b', linewidth=1.2,
                      label='standalone sections'))
    if corrections:
        legend_handles.append(
            mlines.Line2D([], [], color='#e2e8f0', linewidth=1.5,
                          linestyle='--', label='★ z-corrected'))
    ax_corr.legend(handles=legend_handles, fontsize=7,
                   facecolor='#1a2130', edgecolor='#2d3748',
                   labelcolor='#8b9ab0', loc='lower left')

    fig.suptitle('Section z-placement: before vs after correction',
                 color='#e2e8f0', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    return fig


###### QC
def _pick_existing_key(container, candidates):
    for k in candidates:
        if k in container:
            return k
    return None


def _unwrap_da(element):
    da = element
    for _ in range(5):
        if hasattr(da, 'data') and hasattr(da.data, 'shape'):
            return da
        if hasattr(da, 'keys') and callable(da.keys):
            da = da[next(iter(da.keys()))]
        else:
            break
    return da

def _as_pandas(df_like):
    if hasattr(df_like, 'compute'):
        return df_like.compute()
    return df_like


def _resolve_channel_index(da, channel=None):
    """Resolve channel index for a potentially multi-channel image DataArray-like object."""
    if not hasattr(da, 'data'):
        return 0, None

    arr = da.data
    if arr.ndim != 4:
        return 0, None

    # Expected order in this pipeline is typically (c, z, y, x).
    # channel can be int (index) or str (name).
    if channel is None:
        return 0, None

    if isinstance(channel, int):
        idx = int(channel)
        if idx < 0 or idx >= arr.shape[0]:
            raise ValueError(f"channel index {idx} out of range [0, {arr.shape[0]-1}]")
        return idx, None

    if isinstance(channel, str):
        try:
            names = list(get_channel_names(da))
        except Exception:
            names = []
        if len(names) == 0:
            raise ValueError("channel specified by name, but channel names are unavailable for this image")
        if channel not in names:
            raise ValueError(f"channel '{channel}' not found. Available: {names}")
        return int(names.index(channel)), channel

    raise TypeError("channel must be None, int, or str")


def _resolve_lift_slab_bounds(sdata, img_da, n_z, section_n, ref_3d_key_type='dapi_zstack'):
    """
    Determine the full-slab z bounds for a (possibly lifted) 2D-to-3D image element.

    Priority:
      1. Read z_ref_n_planes / z_lift_start_index from img_da.attrs (only available
         in-memory before a zarr write/read round-trip).
      2. Infer from the reference 3D image (dapi_zstack for that section): because
         make_element_3d always places the lifted element at
         z_start = (ref_nz - target_nz) / 2, we can reconstruct it from n_z alone.

    Returns (z_ref_n_planes, z_lift_start, lift_mode, z_slab_lo, z_slab_hi).
    """
    img_attrs = getattr(img_da, 'attrs', {}) or {}

    if img_attrs.get('z_ref_n_planes') is not None:
        # attrs survived (in-memory, not yet round-tripped through zarr)
        z_ref_n_planes = int(img_attrs['z_ref_n_planes'])
        z_lift_start   = float(img_attrs.get('z_lift_start_index', 0.0))
        lift_mode      = img_attrs.get('z_lift_mode', 'reference_slab')
    else:
        # attrs lost after zarr round-trip: infer from dapi_zstack reference
        ref_key = _pick_existing_key(
            sdata.images,
            [
                f'{ref_3d_key_type}-{section_n}' if section_n is not None else None,
                f'{ref_3d_key_type}_{section_n}' if section_n is not None else None,
                ref_3d_key_type,
            ],
        )
        if ref_key is not None:
            ref_da  = _unwrap_da(sdata.images[ref_key])
            ref_arr = ref_da.data
            z_ref_n_planes = int(ref_arr.shape[1]) if ref_arr.ndim == 4 else int(ref_arr.shape[0])
        else:
            z_ref_n_planes = n_z  # no reference found; treat as full slab

        # Reconstruct z_lift_start from the make_element_3d formula
        z_lift_start = (z_ref_n_planes - n_z) / 2.0

        if n_z == z_ref_n_planes:
            lift_mode = 'reference_slab'
        elif n_z == 1:
            lift_mode = 'centered_plane'
        else:
            lift_mode = 'centered_slab'

    # Express slab bounds in *centered* index space where 0 = section midplane.
    # Every lift mode yields the same full-reference-slab bounds:
    #   -(z_ref_n_planes-1)/2  …  +(z_ref_n_planes-1)/2
    # Callers that compute z_idx via mat_cz_inv must convert raw index → centered index by
    # subtracting  z_idx_center_offset = (z_ref_n_planes-1)/2 - z_lift_start
    # before comparing against these bounds.
    center_offset = (z_ref_n_planes - 1) / 2.0
    z_slab_lo = -center_offset
    z_slab_hi = +center_offset
    return z_ref_n_planes, z_lift_start, lift_mode, z_slab_lo, z_slab_hi


def alignment_qc_dashboard(
    sdata,
    section_n=None,
    sections_um=20.0,
    coord_sys='czstack_microns',
    img_key_type='dapi_zstack',
    channel=None,
    y_tolerance=1.0,
    dapi_signal_percentile=1.0,
    thickness_tol_um=1.0,
    max_dapi_pts=120000,
    max_tx_pts=10000,
    overlay_space='index',
    overlay_y_tolerance_idx=0.5,
    max_tx_overlay_pts=20000,
    make_plot=True,
):
    """
    Compact QC for one section (for any image channel):
      1) global z-thickness check (image vs transcripts)
      2) czstack z-overlap check in matched y-slab
      3) transform-only center alignment metric in czstack (median z offset)
      4) XZ overlay in either world (`overlay_space='world'`) or image index (`overlay_space='index'`)
      5) summary metrics panel
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import spatialdata as sd

    if overlay_space not in ('world', 'index'):
        raise ValueError("overlay_space must be 'world' or 'index'")

    # Resolve keys for either per-section or single-section sdata
    img_key = _pick_existing_key(
        sdata.images,
        [
            f'{img_key_type}_{section_n}' if section_n is not None else None,
            f'{img_key_type}-{section_n}' if section_n is not None else None,
            img_key_type,
        ],
    )
    tx_key = _pick_existing_key(
        sdata.points,
        [
            f'transcripts_{section_n}' if section_n is not None else None,
            f'transcripts-{section_n}' if section_n is not None else None,
            'transcripts',
        ],
    )

    if img_key is None:
        raise KeyError(f"Could not find image key for '{img_key_type}' in sdata.images")
    if tx_key is None:
        raise KeyError('Could not find a transcript points key in sdata.points')

    img_da = _unwrap_da(sdata.images[img_key])
    arr = img_da.data

    # Select image channel (if multichannel)
    ch_idx, ch_name = _resolve_channel_index(img_da, channel=channel)
    selected_channel = ch_name if ch_name is not None else ch_idx

    if arr.ndim == 4:
        n_z, n_y, n_x = arr.shape[1], arr.shape[2], arr.shape[3]
        mid_y = n_y // 2
        img_xz = arr[ch_idx, :, mid_y, :].compute().astype(np.float32)
    else:
        n_z, n_y, n_x = arr.shape[0], arr.shape[1], arr.shape[2]
        mid_y = n_y // 2
        img_xz = arr[:, mid_y, :].compute().astype(np.float32)

    # Resolve lift metadata (survives zarr round-trip via dapi_zstack inference)
    _z_ref_n_planes, _z_lift_start, _lift_mode, _z_slab_lo, _z_slab_hi = \
        _resolve_lift_slab_bounds(sdata, img_da, n_z, section_n)

    # Affines
    mat_global = np.array(get_transformation(img_da, to_coordinate_system='global').to_affine_matrix(
        input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')
    ))
    mat_cz = np.array(get_transformation(img_da, to_coordinate_system=coord_sys).to_affine_matrix(
        input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')
    ))
    mat_cz_inv = np.linalg.inv(mat_cz)

    # Image global z support (center-based)
    z_idx = np.arange(n_z, dtype=float)
    x0, y0 = n_x / 2.0, n_y / 2.0
    img_z_global = mat_global[2, 0] * x0 + mat_global[2, 1] * y0 + mat_global[2, 2] * z_idx + mat_global[2, 3]
    img_span_centers = float(img_z_global.max() - img_z_global.min())
    # With mps = sections_depth/(n_z-1), the span between plane-center extremes equals
    # sections_um exactly (plane 0 at -sections_um/2, plane n_z-1 at +sections_um/2).
    img_expected_center_span = float(sections_um)
    img_expected_full_span = float(sections_um)

    # Transcript global support
    tx_global = _as_pandas(sd.transform(sdata.points[tx_key], to_coordinate_system='global'))
    tx_span_global = float(tx_global['z'].max() - tx_global['z'].min())

    pass_img_center = abs(img_span_centers - img_expected_center_span) <= thickness_tol_um
    pass_tx_global = abs(tx_span_global - sections_um) <= thickness_tol_um

    # Build image world XZ at fixed image y-index
    xx, zz = np.meshgrid(np.arange(n_x, dtype=float), np.arange(n_z, dtype=float))
    yy = np.full_like(xx, float(mid_y))
    img_x_cz = mat_cz[0, 0] * xx + mat_cz[0, 1] * yy + mat_cz[0, 2] * zz + mat_cz[0, 3]
    img_z_cz = mat_cz[2, 0] * xx + mat_cz[2, 1] * yy + mat_cz[2, 2] * zz + mat_cz[2, 3]

    y_slice_world = float(
        mat_cz[1, 0] * (n_x / 2.0) + mat_cz[1, 1] * mid_y + mat_cz[1, 2] * (n_z / 2.0) + mat_cz[1, 3]
    )

    flat_i = img_xz.ravel()
    flat_x = img_x_cz.ravel()
    flat_z = img_z_cz.ravel()
    thr = np.percentile(flat_i, dapi_signal_percentile)
    keep_sig = flat_i > thr
    sig_idx = np.flatnonzero(keep_sig)

    img_sig_x = flat_x[keep_sig]
    img_sig_z = flat_z[keep_sig]
    img_sig_i = flat_i[keep_sig]

    if len(img_sig_i) > max_dapi_pts:
        rng = np.random.default_rng(42)
        choose = rng.choice(np.arange(len(img_sig_i)), size=max_dapi_pts, replace=False)
        img_sig_x = img_sig_x[choose]
        img_sig_z = img_sig_z[choose]
        img_sig_i = img_sig_i[choose]
        sig_idx = sig_idx[choose]

    # Transcript czstack support in matched y-slab + image x support (world-based QC metric)
    tx_cz = _as_pandas(sd.transform(sdata.points[tx_key], to_coordinate_system=coord_sys))
    n_tx_total = int(len(tx_cz))
    tx_mask_y = np.ones(n_tx_total, dtype=bool)
    if 'y' in tx_cz.columns:
        tx_mask_y &= np.abs(tx_cz['y'].values - y_slice_world) <= y_tolerance

    tx_mask_x = np.ones(n_tx_total, dtype=bool)
    if len(img_sig_x) > 0:
        tx_mask_x &= (tx_cz['x'].values >= img_sig_x.min()) & (tx_cz['x'].values <= img_sig_x.max())

    tx_mask = tx_mask_y & tx_mask_x
    tx_slab = tx_cz.loc[tx_mask].copy()
    n_tx_after_y = int(tx_mask_y.sum())

    if len(tx_slab) > max_tx_pts:
        tx_slab = tx_slab.sample(max_tx_pts, random_state=42)

    img_cz_rng = (float(np.min(img_sig_z)), float(np.max(img_sig_z))) if len(img_sig_z) else (np.nan, np.nan)
    tx_cz_rng = (float(tx_slab['z'].min()), float(tx_slab['z'].max())) if len(tx_slab) else (np.nan, np.nan)

    if len(img_sig_z) > 10 and len(tx_slab) > 10:
        i_lo, i_hi = np.percentile(img_sig_z, [0.5, 99.5])
        t_lo, t_hi = np.percentile(tx_slab['z'].values, [0.5, 99.5])
        overlap = max(0.0, min(i_hi, t_hi) - max(i_lo, t_lo))
        union = max(i_hi, t_hi) - min(i_lo, t_lo)
        overlap_frac = float(overlap / union) if union > 0 else np.nan

        i_med = float(np.median(img_sig_z))
        t_med = float(np.median(tx_slab['z'].values))
        median_z_offset_cz = float(t_med - i_med)

        i_q25, i_q75 = np.percentile(img_sig_z, [25, 75])
        i_iqr = max(float(i_q75 - i_q25), 1e-6)
        median_z_offset_iqr_norm = float(median_z_offset_cz / i_iqr)
    else:
        overlap_frac = np.nan
        median_z_offset_cz = np.nan
        median_z_offset_iqr_norm = np.nan

    # Overlay-specific transcript selection (to prevent opposite tilt artifacts)
    if overlay_space == 'world':
        overlay_img_x = img_sig_x
        overlay_img_z = img_sig_z
        overlay_img_i = img_sig_i
        overlay_tx_x = tx_slab['x'].values if len(tx_slab) else np.array([])
        overlay_tx_z = tx_slab['z'].values if len(tx_slab) else np.array([])
        n_tx_overlay = int(len(overlay_tx_x))
        overlay_desc = f"world y-slab ±{y_tolerance}"
    else:
        # Image points in index XZ (centered: 0 = section midplane)
        x_idx_flat = xx.ravel()
        z_idx_flat = zz.ravel()
        overlay_img_x = x_idx_flat[sig_idx]
        overlay_img_z = z_idx_flat[sig_idx]  # will be centered below after _z_idx_center_offset is known
        overlay_img_i = flat_i[sig_idx]

        # Transform a sampled set of transcripts to image index space and filter by y-index
        if len(tx_cz) > 300000:
            tx_overlay_df = tx_cz.sample(300000, random_state=42)
        else:
            tx_overlay_df = tx_cz

        xyz = tx_overlay_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        xyz1 = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
        idx_pts = xyz1 @ mat_cz_inv.T
        tx_xi = idx_pts[:, 0]
        tx_yi = idx_pts[:, 1]
        tx_zi = idx_pts[:, 2]

        # Convert raw image-index z to centered-index space (0 = section midplane).
        _z_idx_center_offset = (_z_ref_n_planes - 1) / 2.0 - _z_lift_start
        tx_zi = tx_zi - _z_idx_center_offset
        overlay_img_z = overlay_img_z - _z_idx_center_offset  # center image z indices for plotting

        mask_i = np.abs(tx_yi - mid_y) <= overlay_y_tolerance_idx
        mask_i &= (tx_xi >= 0) & (tx_xi <= (n_x - 1))
        mask_i &= (tx_zi >= _z_slab_lo) & (tx_zi <= _z_slab_hi)

        overlay_tx_x = tx_xi[mask_i]
        overlay_tx_z = tx_zi[mask_i]

        if len(overlay_tx_x) > max_tx_overlay_pts:
            rng = np.random.default_rng(42)
            keep = rng.choice(np.arange(len(overlay_tx_x)), size=max_tx_overlay_pts, replace=False)
            overlay_tx_x = overlay_tx_x[keep]
            overlay_tx_z = overlay_tx_z[keep]

        n_tx_overlay = int(len(overlay_tx_x))
        overlay_desc = f"index y-slab ±{overlay_y_tolerance_idx}"

    metrics = {
        'section_n': section_n,
        'img_key_type': img_key_type,
        'img_key': img_key,
        'channel': selected_channel,
        'tx_key': tx_key,
        'sections_um_target': float(sections_um),
        'n_z_planes': int(n_z),
        'img_expected_center_span_um': float(img_expected_center_span),
        'img_expected_full_span_um': float(img_expected_full_span),
        'img_global_span_um': float(img_span_centers),
        'tx_global_span_um': float(tx_span_global),
        'img_center_span_pass': bool(pass_img_center),
        'tx_global_span_pass': bool(pass_tx_global),
        'coord_sys': coord_sys,
        'img_cz_range': img_cz_rng,
        'tx_cz_range': tx_cz_rng,
        'overlap_fraction': overlap_frac,
        'median_z_offset_cz': median_z_offset_cz,
        'median_z_offset_iqr_norm': median_z_offset_iqr_norm,
        'n_tx_total': n_tx_total,
        'n_tx_after_y': n_tx_after_y,
        'n_tx_in_slab': int(len(tx_slab)),
        'overlay_space': overlay_space,
        'n_tx_overlay': n_tx_overlay,
        'y_tolerance': float(y_tolerance),
    }

    if make_plot:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        label_img = f"{img_key_type}[{selected_channel}]" if arr.ndim == 4 else img_key_type

        # (A) Global thickness check — center both distributions for a span-only comparison
        # (absolute z positions differ because DAPI global transform was shifted by center_z=True
        # while transcript native z may have a different z-reference in global space)
        ax = axes[0, 0]
        img_z_c = img_z_global - img_z_global.mean()
        tx_z_c  = tx_global['z'].values - tx_global['z'].values.mean()
        ax.hist(img_z_c, bins=min(30, n_z), alpha=0.6, density=True, label=f'{label_img} (global, centered)')
        ax.hist(tx_z_c, bins=40, alpha=0.5, density=True, label='TX (global, centered)')
        ax.axvline( img_expected_center_span / 2, color='tab:blue',   linestyle='--', linewidth=1, label=f'img span ={img_expected_center_span:.1f}µm')
        ax.axvline(-img_expected_center_span / 2, color='tab:blue',   linestyle='--', linewidth=1)
        ax.axvline( sections_um / 2,              color='tab:orange', linestyle='--', linewidth=1, label=f'TX span ={sections_um:.1f}µm')
        ax.axvline(-sections_um / 2,              color='tab:orange', linestyle='--', linewidth=1)
        ax.set_title('Global z thickness (centered)')
        ax.set_xlabel('z − mean(z)  (µm)')
        ax.set_ylabel('density')
        ax.legend(fontsize=8)

        # (B) czstack overlap histogram
        ax = axes[0, 1]
        if len(img_sig_z) and len(tx_slab):
            bins = np.linspace(min(np.min(img_sig_z), tx_slab['z'].min()),
                               max(np.max(img_sig_z), tx_slab['z'].max()), 45)
            ax.hist(img_sig_z, bins=bins, alpha=0.55, density=True, label=f'{label_img} signal')
            ax.hist(tx_slab['z'].values, bins=bins, alpha=0.55, density=True, label='TX slab')
        ax.set_title(f'czstack z overlap (y±{y_tolerance})')
        ax.set_xlabel(f'z ({coord_sys})')
        ax.set_ylabel('density')
        ax.legend(fontsize=8)

        # (C) XZ overlay
        ax = axes[1, 0]
        if len(overlay_img_z):
            ax.scatter(overlay_img_x, overlay_img_z, c=overlay_img_i, s=1, cmap='magma', alpha=0.35, linewidths=0)
        if len(overlay_tx_z):
            ax.scatter(overlay_tx_x, overlay_tx_z, s=3, alpha=0.6, c='cyan', linewidths=0)
        if overlay_space == 'index':
            ax.set_title('XZ overlay in image index space')
            ax.set_xlabel('x (image index)')
            ax.set_ylabel('z (image index)')
        else:
            ax.set_title('XZ overlay in czstack world')
            ax.set_xlabel(f'x ({coord_sys})')
            ax.set_ylabel(f'z ({coord_sys})')

        # (D) Text summary
        ax = axes[1, 1]
        ax.axis('off')
        _lift_summary = (
            f"{_lift_mode}  ref_nz={_z_ref_n_planes}  start={_z_lift_start:.1f}\n"
            f"Slab idx bounds: [{_z_slab_lo:.1f}, {_z_slab_hi:.1f}]\n"
        ) if _lift_mode != 'reference_slab' else f"{_lift_mode}\n"
        txt = (
            f"Section: {section_n if section_n is not None else 'single'}\n"
            f"Image key: {img_key}\nChannel: {selected_channel}\nTX key: {tx_key}\n"
            f"Lift: {_lift_summary}\n"
            f"Image global span: {img_span_centers:.3f} µm\n"
            f"Image expected (center-span): {img_expected_center_span:.3f} µm\n"
            f"TX global span: {tx_span_global:.3f} µm\n"
            f"TX expected: {sections_um:.3f} µm\n\n"
            f"PASS image center-span: {pass_img_center}\n"
            f"PASS TX span: {pass_tx_global}\n\n"
            f"Image cz range: [{img_cz_rng[0]:.2f}, {img_cz_rng[1]:.2f}]\n"
            f"TX cz range: [{tx_cz_rng[0]:.2f}, {tx_cz_rng[1]:.2f}]\n"
            f"Overlap fraction: {overlap_frac:.3f}\n"
            f"Median z offset (TX-image): {median_z_offset_cz:.3f}\n"
            f"Offset / image IQR: {median_z_offset_iqr_norm:.3f}\n"
            f"Overlay space: {overlay_space} ({overlay_desc})\n"
            f"n TX total: {n_tx_total}\n"
            f"n TX after y-filter: {n_tx_after_y}\n"
            f"n TX in slab: {len(tx_slab)}\n"
            f"n TX in overlay: {n_tx_overlay}"
        )
        ax.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10, family='monospace')

        plt.tight_layout()
        plt.show()

    return metrics

def diagnose_transcript_z_within_section(
    sdata,
    section_n,
    img_key_type='rna',
    coord_sys='czstack_microns',
):
    """
    Check whether transcripts for a section lie within the section's z slab.

    Correctly handles 'centered_plane' and 'centered_slab' lift modes: the out-of-slab
    count is measured against the FULL REFERENCE SLAB bounds (not the image's n_z planes),
    so a single-plane centered image does not report 99%+ out of bounds.
    """
    import numpy as np
    import spatialdata as sd

    # Resolve keys
    img_key = None
    for k in [f'{img_key_type}-{section_n}', f'{img_key_type}_{section_n}', img_key_type]:
        if k in sdata.images:
            img_key = k
            break
    tx_key = None
    for k in [f'transcripts-{section_n}', f'transcripts_{section_n}', 'transcripts']:
        if k in sdata.points:
            tx_key = k
            break

    if img_key is None or tx_key is None:
        raise KeyError(f'Could not resolve keys. img_key={img_key}, tx_key={tx_key}')

    img_da = _unwrap_da(sdata.images[img_key])
    arr = img_da.data
    if arr.ndim == 4:
        n_z = int(arr.shape[1])
    elif arr.ndim == 3:
        n_z = int(arr.shape[0])
    else:
        raise ValueError(f'Unsupported image ndim={arr.ndim}')

    # Resolve lift metadata (survives zarr round-trip via dapi_zstack inference)
    z_ref_n_planes, z_lift_start, lift_mode, z_slab_lo, z_slab_hi = \
        _resolve_lift_slab_bounds(sdata, img_da, n_z, section_n)

    mat = np.array(
        get_transformation(img_da, to_coordinate_system=coord_sys).to_affine_matrix(
            input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')
        ),
        dtype=np.float64,
    )
    mat_inv = np.linalg.inv(mat)

    tx = sd.transform(sdata.points[tx_key], to_coordinate_system=coord_sys)
    tx_pdf = tx.compute() if hasattr(tx, 'compute') else tx

    xyz = tx_pdf[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    xyz1 = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    idx = xyz1 @ mat_inv.T
    z_idx = idx[:, 2]

    # Convert raw image-index z to centered-index space (0 = section midplane).
    # Raw z_idx from mat_inv includes the lift offset; subtracting z_idx_center_offset
    # aligns it with the centered slab bounds from _resolve_lift_slab_bounds.
    _z_idx_center_offset = (z_ref_n_planes - 1) / 2.0 - z_lift_start
    z_idx = z_idx - _z_idx_center_offset

    # Out-of-bounds relative to the full reference slab, not just this image's n_z planes.
    out_low  = int(np.sum(z_idx < z_slab_lo))
    out_high = int(np.sum(z_idx > z_slab_hi))
    out_any  = out_low + out_high
    frac_out = float(out_any / len(z_idx)) if len(z_idx) else np.nan

    print(f'Section {section_n}  | image={img_key}  tx={tx_key}')
    print(f'n_tx={len(z_idx)}  n_z_image={n_z}  lift_mode={lift_mode}')
    print(f'z_ref_n_planes={z_ref_n_planes}  z_lift_start={z_lift_start:.1f}')
    print(f'Slab bounds (centered idx space): [{z_slab_lo:.1f}, {z_slab_hi:.1f}]')
    print(f'z_idx min/max: {z_idx.min():.3f} / {z_idx.max():.3f}')
    print(f'out_of_slab: {out_any} ({frac_out:.3%})  [low={out_low}, high={out_high}]')

    return {
        'section_n': section_n,
        'img_key': img_key,
        'tx_key': tx_key,
        'n_tx': int(len(z_idx)),
        'n_z_image': int(n_z),
        'lift_mode': lift_mode,
        'z_ref_n_planes': int(z_ref_n_planes),
        'z_lift_start': float(z_lift_start),
        'z_slab_lo': float(z_slab_lo),
        'z_slab_hi': float(z_slab_hi),
        'z_idx_min': float(z_idx.min()),
        'z_idx_max': float(z_idx.max()),
        'out_of_slab_n': int(out_any),
        'out_of_slab_fraction': float(frac_out),
    }


def alignment_qc_summary_by_section(
    combined_data,
    sections,
    sections_um=20.0,
    img_key_type='dapi_zstack',
    channel=None,
    coord_sys='czstack_microns',
    y_tolerance=1.0,
):
    """Run compact QC across sections and plot a concise summary."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []
    for s_n in sections:
        try:
            m = alignment_qc_dashboard(
                combined_data,
                section_n=s_n,
                sections_um=sections_um,
                img_key_type=img_key_type,
                channel=channel,
                coord_sys=coord_sys,
                y_tolerance=y_tolerance,
                make_plot=False,
            )
            rows.append(m)
        except Exception as e:
            rows.append({'section_n': s_n, 'error': str(e)})

    df = pd.DataFrame(rows).sort_values('section_n').reset_index(drop=True)

    ok = df[~df.get('error', pd.Series([None] * len(df))).notna()].copy()
    if len(ok) == 0:
        display(df)
        return df

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))

    # 1) Thickness spans
    axes[0].plot(ok['section_n'], ok['img_global_span_um'], 'o-', label='Image global span')
    axes[0].plot(ok['section_n'], ok['tx_global_span_um'], 'o-', label='TX global span')
    axes[0].axhline(sections_um, linestyle='--', color='k', linewidth=1, label='20 µm target')
    axes[0].set_title('Global thickness by section')
    axes[0].set_xlabel('section')
    axes[0].set_ylabel('span (µm)')
    axes[0].legend(fontsize=8)

    # 2) Overlap fraction
    axes[1].bar(ok['section_n'].astype(str), ok['overlap_fraction'])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('czstack z overlap fraction')
    axes[1].set_xlabel('section')
    axes[1].set_ylabel('overlap (0-1)')

    # 3) transform-only metric (center offset)
    axes[2].axhline(0.0, linestyle='--', color='k', linewidth=1)
    axes[2].plot(ok['section_n'], ok['median_z_offset_cz'], 'o-', color='tab:green')
    axes[2].set_title('Median z offset (TX - image)')
    axes[2].set_xlabel('section')
    axes[2].set_ylabel(f'offset ({coord_sys} z units)')

    # 4) czstack ranges
    i_mins = ok['img_cz_range'].apply(lambda x: x[0])
    i_maxs = ok['img_cz_range'].apply(lambda x: x[1])
    t_mins = ok['tx_cz_range'].apply(lambda x: x[0])
    t_maxs = ok['tx_cz_range'].apply(lambda x: x[1])

    y = np.arange(len(ok))
    axes[3].hlines(y + 0.12, i_mins, i_maxs, linewidth=3, label='Image')
    axes[3].hlines(y - 0.12, t_mins, t_maxs, linewidth=3, label='TX')
    axes[3].set_yticks(y)
    axes[3].set_yticklabels(ok['section_n'].astype(str))
    axes[3].set_title(f'czstack z-ranges ({coord_sys})')
    axes[3].set_xlabel('z')
    axes[3].set_ylabel('section')
    axes[3].legend(fontsize=8)

    plt.suptitle(f"QC summary: img_key_type={img_key_type}, channel={channel}", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()
    return df
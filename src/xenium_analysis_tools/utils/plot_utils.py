def get_vals_perc(img, chan, vmin_val=None, vmax_val=None, vmin_perc=None, vmax_perc=None, res_level=4):
    if res_level > len(sd.get_pyramid_levels(img))-1:
        res_level = len(sd.get_pyramid_levels(img))-1
        print(f'Using resolution level: {res_level}')
    if vmin_perc is not None or vmax_perc is not None:
        ch_vals = sd.get_pyramid_levels(img, n=res_level).sel(c=chan)
    vmin = None
    if vmin_val is not None:
        vmin = vmin_val
    elif vmin_perc is not None:
        vmin = np.percentile(ch_vals.values, vmin_perc)

    vmax = None
    if vmax_val is not None:
        vmax = vmax_val
    elif vmax_perc is not None:
        vmax = np.percentile(ch_vals.values, vmax_perc)

    return vmin, vmax


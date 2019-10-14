import matplotlib
from matplotlib import ticker
from matplotlib.axis import YAxis
import math
import numpy as np


class SciFuncFormatter(ticker.Formatter):
    # To put full sci notation into each axis label rather than split offsetText

    sFormatter = ticker.ScalarFormatter(useOffset=False, useMathText=True)

    def __call__(self, x, pos=None):
        return "${}$".format(SciFuncFormatter.sFormatter._formatSciNotation('%.10e' % x))

    def format_data(self, value):
        # e.g. for the navigation toolbar, no latex
        return '%-8g' % value


class BoundedMaxNLocator(ticker.MaxNLocator):
    # Tick locator class that only returns ticks within bounds, and if pruned, pruned not to overlap ends of axes
    # Also tries to correct default (simple x3 heuristic) for long tick labels

    def __init__(self, nbins='auto', prune=True, **kwargs):
        self.bounded_prune = prune
        super(BoundedMaxNLocator, self).__init__(nbins=nbins, **kwargs)

    def _bounded_prune(self, locs, vmin, vmax, label_len):
        if len(locs) > 1 and self.bounded_prune:
            if locs[0] - vmin < label_len * 0.5:
                locs = locs[1:]
            if vmax - locs[-1] < label_len * 0.5 and len(locs) > 1:
                locs = locs[:-1]
        return locs

    def tick_values(self, vmin, vmax):
        # Max N locator will produce locations outside vmin, vmax, so even if pruned
        # there can be points very close to the actual bounds. Let's cut them out.
        # Also account for tick labels with aspect ratio > 3 (default often-violated heuristic)
        # - use better heuristic based on number of characters in label and typical font aspect ratio

        _preferred_steps = None

        axes = self.axis.axes
        tick = self.axis._get_tick(True)
        rotation = tick._labelrotation[1]

        if isinstance(self.axis, YAxis):
            rotation += 90
            ends = axes.transAxes.transform([[0, 0], [0, 1]])
            length = ((ends[1][1] - ends[0][1]) / axes.figure.dpi) * 72
        else:
            ends = axes.transAxes.transform([[0, 0], [1, 0]])
            length = ((ends[1][0] - ends[0][0]) / axes.figure.dpi) * 72
        size_ratio = tick.label1.get_size() / length
        cos_rotation = abs(math.cos(math.radians(rotation)))
        font_aspect = 0.65 * cos_rotation
        formatter = self.axis.major.formatter

        # first guess
        if cos_rotation > 0.05:
            label_len = size_ratio * 1.5 * (vmax - vmin)
            label_space = label_len * 1.1
        else:
            # text orthogonal to axis
            label_len = size_ratio * 1.35 * (vmax - vmin)
            label_space = label_len * 1.25

        delta = label_len / 2 if self.bounded_prune else 0

        try:
            nbins = int((vmax - vmin - 2 * delta) / label_space) + 1
            if nbins > 4:
                # use more space for ticks
                _nbins = int((vmax - vmin - 2 * delta) / ((1.5 if nbins > 6 else 1.3) * label_space)) + 1
            min_n_ticks = min(nbins, 3)
            if self._nbins != 'auto':
                nbins = min(self._nbins, nbins)
            while True:
                locs = self._spaced_ticks(vmin + delta, vmax - delta, label_len * 1.1, min_n_ticks, nbins)
                if len(locs) or min_n_ticks == 1:
                    break
                min_n_ticks -= 1
            if cos_rotation > 0.05 and isinstance(formatter, ticker.ScalarFormatter) and len(locs) > 1:

                def _get_Label_len():
                    if not len(locs):
                        return 0
                    formatter.set_locs(locs)
                    # get non-latex version of label
                    form = formatter.format
                    i = form.index('%')
                    i2 = form.index('f', i)
                    label = form[i:i2 + 1] % locs[0]
                    char_len = len(label)
                    if '.' in label:
                        char_len -= 0.4
                    if len(locs) > 1:
                        label = form[i:i2 + 1] % locs[-1]
                        char_len2 = len(label)
                        if '.' in label:
                            char_len2 -= 0.4
                        char_len = max(char_len, char_len2)

                    return size_ratio * max(2.0, char_len * font_aspect) * (vmax - vmin)

                label_len = _get_Label_len()
                locs = self._bounded_prune(locs, vmin, vmax, label_len)
                if len(locs) < 3 or locs[1] - locs[0] < label_len * (1.1 if len(locs) < 4 else 1.5) \
                        or (locs[0] - vmin > label_len * 1.5 or vmax - locs[-1] > label_len * 1.5):
                    # check for long labels not accounted for the the current "*3" aspect ratio heuristic for labels
                    # and labels that are too tightly spaced
                    delta = label_len / 2 if self.bounded_prune else 0
                    for fac in [1.5, 1.35, 1.1]:
                        nbins = int((vmax - vmin - 2 * delta) / (fac * label_len)) + 1
                        if nbins >= 4:
                            break
                    if self._nbins != 'auto':
                        nbins = min(self._nbins, nbins)
                    min_n_ticks = min(min_n_ticks, nbins)
                    retry = True
                    try_shorter = True
                    while min_n_ticks:
                        locs = self._spaced_ticks(vmin + delta, vmax - delta, label_len * 1.1, min_n_ticks, nbins)
                        if len(locs):
                            new_len = _get_Label_len()
                            if not np.isclose(new_len, label_len):
                                label_len = new_len
                                delta = label_len / 2 if self.bounded_prune else 0
                                locs = self._bounded_prune(locs, vmin, vmax, label_len)
                                if retry:
                                    retry = False
                                    continue
                        elif min_n_ticks > 1 and try_shorter:
                            # Original label length may be too long for good ticks which exist
                            delta /= 2
                            label_len /= 2
                            try_shorter = False
                            locs = self._spaced_ticks(vmin + delta, vmax - delta, label_len * 1.1, min_n_ticks, nbins)
                            if len(locs):
                                label_len = _get_Label_len()
                                delta = label_len / 2 if self.bounded_prune else 0
                                continue

                        if len(locs) < 2 and _preferred_steps is None \
                                and label_len < (vmax - vmin) / (2 if min_n_ticks > 1 else 1.05):
                            _preferred_steps = self._steps
                            min_n_ticks = min(min_n_ticks, 2)
                            self.set_params(steps=[1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10])
                            continue
                        if min_n_ticks == 1 and len(locs) == 1 or len(locs) >= min_n_ticks > 1 \
                                and locs[1] - locs[0] > _get_Label_len() * 1.1:
                            break
                        min_n_ticks -= 1
                    if len(locs) <= 1 and size_ratio * font_aspect < 0.9:
                        # if no ticks, check for short integer number location in the range that may have been missed
                        # because adding any other values would make label length much longer
                        scale, offset = ticker.scale_range(vmin, vmax, 1)
                        loc = round((vmin + vmax) / (2 * scale)) * scale
                        if vmin < loc < vmax:
                            locs = [loc]
                            label_len = _get_Label_len()
                            locs = self._bounded_prune(locs, vmin, vmax, label_len)
            else:
                locs = self._bounded_prune(locs, vmin, vmax, label_len)
        finally:
            if _preferred_steps is not None:
                self.set_params(steps=_preferred_steps)

        return locs

    def _spaced_ticks(self, vmin, vmax, label_len, min_ticks, nbins):

        scale, offset = ticker.scale_range(vmin, vmax, nbins)
        _vmin = vmin - offset
        _vmax = vmax - offset
        raw_step = max(label_len, (_vmax - _vmin) / max(1, nbins - 1))
        steps = self._extended_steps * scale

        istep = np.nonzero(steps >= (raw_step if nbins > 1 else ((_vmax - _vmin) / 2)))[0]
        if len(istep) == 0:
            if steps[-1] < label_len:
                return []
            istep = len(steps) - 1
        else:
            istep = istep[0]

        # This is an upper limit; move to smaller or half-phase steps if necessary.
        for istep in reversed(range(istep + 1)):
            step = steps[istep]

            if step < label_len:
                # e.g. instead of -x 0 x, also try -x/2, x/2
                best_vmin = (_vmin // step + 1) * step
                step *= 2
                if step < label_len:
                    return []
            else:
                best_vmin = (_vmin // step) * step

            low = _ge(_vmin - best_vmin, offset, step)
            high = _le(_vmax - best_vmin, offset, step)
            if high - low + 1 >= min_ticks:
                return np.arange(low, high + 1) * step + (best_vmin + offset)

        return []


def _closeto(ms, edge, offset, step):
    if offset > 0:
        digits = np.log10(offset / step)
        tol = max(1e-10, 10 ** (digits - 12))
        tol = min(0.4999, tol)
    else:
        tol = 1e-10
    return abs(ms - edge) < tol


def _le(x, offset, step):
    """Return the largest n: n*step <= x."""
    d, m = divmod(x, step)
    if _closeto(m / step, 1, abs(offset), step):
        return d + 1
    return d


def _ge(x, offset, step):
    """Return the smallest n: n*step >= x."""
    d, m = divmod(x, step)
    if _closeto(m / step, 0, abs(offset), step):
        return d
    return d + 1

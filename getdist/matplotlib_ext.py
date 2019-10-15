import matplotlib
from matplotlib import ticker
from matplotlib.axis import YAxis
import math
import numpy as np
from bisect import bisect_left


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

    def __init__(self, nbins='auto', prune=True, step_groups=[[1, 2, 5, 10], [2.5], [3, 4, 6, 8], [1.5, 7, 9]]):
        self.bounded_prune = prune
        self._step_groups = [_staircase(np.array(steps)) for steps in step_groups]
        self._offsets = []
        for g in step_groups:
            g2 = []
            for x in g:
                if x % 2 < 1e-6:
                    g2.append(x // 2)
                else:
                    g2.append(0)
            self._offsets.append(_staircase(np.array(g2)))
        super(BoundedMaxNLocator, self).__init__(nbins=nbins)

    def _bounded_prune(self, locs, label_len):
        if len(locs) > 1 and self.bounded_prune:
            if locs[0] - self._range[0] < label_len * 0.5:
                locs = locs[1:]
            if self._range[1] - locs[-1] < label_len * 0.5 and len(locs) > 1:
                locs = locs[:-1]
        return locs

    def _get_label_len(self, locs):
        if not len(locs):
            return 0
        self._formatter.set_locs(locs)
        # get non-latex version of label
        form = self._formatter.format
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

        return max(2.0, char_len * self._font_aspect) * self._char_size_scale

    def tick_values(self, vmin, vmax):
        # Max N locator will produce locations outside vmin, vmax, so even if pruned
        # there can be points very close to the actual bounds. Let's cut them out.
        # Also account for tick labels with aspect ratio > 3 (default often-violated heuristic)
        # - use better heuristic based on number of characters in label and typical font aspect ratio

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
        self._font_aspect = 0.65 * cos_rotation
        self._char_size_scale = size_ratio * (vmax - vmin)
        self._formatter = self.axis.major.formatter
        self._range = (vmin, vmax)

        # first guess
        if cos_rotation > 0.05:
            label_len = size_ratio * 1.5 * (vmax - vmin)
            label_space = label_len * 1.1
        else:
            # text orthogonal to axis
            label_len = size_ratio * 1.35 * (vmax - vmin)
            label_space = label_len * 1.25

        delta = label_len / 2 if self.bounded_prune else 0
        nbins = int((vmax - vmin - 2 * delta) / label_space) + 1
        if nbins > 4:
            # use more space for ticks
            _nbins = int((vmax - vmin - 2 * delta) / ((1.5 if nbins > 6 else 1.3) * label_space)) + 1
        min_n_ticks = min(nbins, 2)
        nbins = min(self._nbins if self._nbins != 'auto' else 9, nbins)
        while True:
            locs = self._spaced_ticks(vmin + delta, vmax - delta, label_len, min_n_ticks, nbins, False)
            if len(locs) or min_n_ticks == 1:
                break
            min_n_ticks -= 1
        if cos_rotation > 0.05 and isinstance(self._formatter, ticker.ScalarFormatter) and len(locs) > 1:

            label_len = self._get_label_len(locs)
            locs = self._bounded_prune(locs, label_len)
            if len(locs) > 1:
                step = locs[1] - locs[0]
            if len(locs) < 3 or step < label_len * (1.1 if len(locs) < 4 else 1.5) \
                    or (locs[0] - vmin > min(step, label_len * 1.5) or vmax - locs[-1] > min(step, label_len * 1.5)):
                # check for long labels, labels that are too tightly spaced, or large tick-free gaps at axes ends
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
                    locs = self._spaced_ticks(vmin + delta, vmax - delta, label_len, min_n_ticks, nbins)
                    if len(locs):
                        new_len = self._get_label_len(locs)
                        if not np.isclose(new_len, label_len):
                            label_len = new_len
                            delta = label_len / 2 if self.bounded_prune else 0
                            if retry:
                                retry = False
                                continue
                            locs = self._bounded_prune(locs, label_len)
                    elif min_n_ticks > 1 and try_shorter:
                        # Original label length may be too long for good ticks which exist
                        delta /= 2
                        label_len /= 2
                        try_shorter = False
                        locs = self._spaced_ticks(vmin + delta, vmax - delta, label_len, min_n_ticks, nbins)
                        if len(locs):
                            label_len = self._get_label_len(locs)
                            delta = label_len / 2 if self.bounded_prune else 0
                            continue

                    if min_n_ticks == 1 and len(locs) == 1 or len(locs) >= min_n_ticks > 1 \
                            and locs[1] - locs[0] > self._get_label_len(locs) * 1.1:
                        break
                    min_n_ticks -= 1
                    locs = []
                if len(locs) <= 1 and size_ratio * self._font_aspect < 0.9:
                    # if no ticks, check for short integer number location in the range that may have been missed
                    # because adding any other values would make label length much longer
                    scale, offset = ticker.scale_range(vmin, vmax, 1)
                    loc = round((vmin + vmax) / (2 * scale)) * scale
                    if vmin < loc < vmax:
                        locs = [loc]
                        label_len = self._get_label_len(locs)
                        return self._bounded_prune(locs, label_len)
        else:
            return self._bounded_prune(locs, label_len)

        return locs

    def _valid(self, locs):
        label_len = self._get_label_len(locs)
        return (len(locs) < 2 or locs[1] - locs[0] > label_len * 1.1) and (locs[0] - self._range[0] > label_len / 2) and \
               (self._range[1] - locs[-1] > label_len / 2)

    def _spaced_ticks(self, vmin, vmax, _label_len, min_ticks, nbins, check_ends=True):

        scale, offset = ticker.scale_range(vmin, vmax, nbins)
        _vmin = vmin - offset
        _vmax = vmax - offset
        _range = _vmax - _vmin
        round_center = round((_vmin + _vmax) / (20 * scale)) * 10 * scale
        label_len = _label_len * 1.1
        raw_step = max(label_len, _range / max(1, nbins - 1))
        raw_step1 = _range / max(1, nbins)
        best = []
        for step_ix, (_steps, _offsets) in enumerate(zip(self._step_groups, self._offsets)):

            steps = _steps * scale

            istep = min(len(steps) - 1, bisect_left(steps, raw_step if nbins > 1 else (_range / 2)))
            if not istep:
                continue
            # This is an upper limit; move to smaller or half-phase steps if necessary.
            for off in [False, True]:
                for i in reversed(range(istep + 1)):
                    if off and not _offsets[i]:
                        continue
                    step = steps[i]
                    if step < label_len:
                        break

                    best_vmin = (_vmin // step) * step
                    if step_ix:
                        # For less nice steps, try to make them hit any round numbers in range
                        if _vmin < round_center < _vmax:
                            rem = (round((round_center - best_vmin) / scale) * scale) % step
                            best_vmin += rem

                    if off:
                        # try half-offset steps, e.g. to get -x/2, x/2 as well as -x,0,x
                        best_vmin += scale * _offsets[i]

                    low = _ge(_vmin - best_vmin, offset, step)
                    high = _le(_vmax - best_vmin, offset, step)
                    if min_ticks <= high - low + 1 <= nbins:
                        ticks = np.arange(low, high + 1) * step + (best_vmin + offset)
                        if step_ix and ((not len(best) or len(ticks) < len(best)) \
                                        and step > raw_step1 and step > label_len * 1.5
                                        or step < label_len * 1.3) or check_ends and min_ticks > 1 and (
                                ticks[0] - self._range[0] > max(min(_range / 3, step), label_len) * 1.1
                                or self._range[1] - ticks[-1] > max(min(_range / 3, step), label_len) * 1.1):
                            # prefer spacing where some ticks nearish the ends and ticks not too close in centre
                            if not len(best) and self._valid(ticks):
                                best = ticks
                        else:

                            parts = np.round(ticks / scale).astype(np.int)
                            if not len(best) or len(best) <= min(3, len(ticks)) \
                                    and not np.all(np.remainder(parts, 10)) and (not check_ends or self._valid(ticks)):
                                return ticks

        return best


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


def _staircase(steps):
    if len(steps) == 1:
        return np.array([0.1 * steps[0], steps[0]])
    else:
        flights = (0.1 * steps[:-1], steps, 10 * steps[1])
        return np.hstack(flights)

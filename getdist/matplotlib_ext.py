import matplotlib
from matplotlib import ticker
from matplotlib.axis import YAxis
import math


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

        _nbins = self._nbins
        _min_ticks = self._min_n_ticks

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
        label_len = size_ratio * 1.5 * (vmax - vmin)
        delta = label_len / 2 if self.bounded_prune else 0

        try:
            self._nbins = int((vmax - vmin - 2 * delta) / (1.1 * label_len)) + 1
            if self._nbins > 4:
                # use more space for ticks
                self._nbins = int((vmax - vmin - 2 * delta) / (1.35 * label_len)) + 1
                self._min_n_ticks = 3
            if _nbins != 'auto':
                self._nbins = min(self._nbins, _nbins)

            locs = super(BoundedMaxNLocator, self).tick_values(vmin + delta, vmax - delta)
            locs = [x for x in locs if vmin <= x <= vmax]

            if cos_rotation > 0.05 and isinstance(formatter, ticker.ScalarFormatter) and len(locs) > 1:

                def _get_Label_len():
                    formatter.set_locs(locs)
                    # get non-latex version of label
                    form = formatter.format
                    i = form.index('%')
                    i2 = form.index('f', i)
                    label = form[i:i2 + 1] % locs[0]
                    char_len = len(label)
                    if '.' in label:
                        char_len -= 0.4
                    return size_ratio * max(2.0, char_len * font_aspect) * (vmax - vmin)

                label_len = _get_Label_len()
                if locs[1] - locs[0] < label_len * 1.1 or len(locs) != 3:
                    # check for long labels not accounted for the the current "*3" aspect ratio heuristic for labels
                    # and labels too tightly spaced
                    delta = label_len / 2 if self.bounded_prune else 0
                    self._nbins = int((vmax - vmin - 2 * delta) / (1.1 * label_len)) + 1
                    if self._nbins > 3:
                        self._nbins = max(3, int((vmax - vmin - 2 * delta) / (1.35 * label_len)) + 1)
                    while self._nbins:
                        self._min_n_ticks = min(self._min_n_ticks, self._nbins)
                        locs = super(BoundedMaxNLocator, self).tick_values(vmin + delta, vmax - delta)
                        locs = [x for x in locs if vmin <= x <= vmax]
                        label_len = _get_Label_len()
                        locs = self._bounded_prune(locs, vmin, vmax, label_len)
                        if len(locs) < 2 or locs[1] - locs[0] > label_len * 1.1:
                            break
                        self._nbins = self._nbins - 1
                else:
                    locs = self._bounded_prune(locs, vmin, vmax, label_len)
            else:
                locs = self._bounded_prune(locs, vmin, vmax, label_len)
        finally:
            self._nbins = _nbins
            self._min_n_ticks = _min_ticks

        return locs

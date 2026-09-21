"""Map viewport arithmetic, independent of GUI and algorithm state."""

from dataclasses import dataclass
from math import isfinite

FIT_VIEW = (0.0, 0.0, 1.0, 1.0)
MAX_ZOOM = 32.0


@dataclass
class MapViewport:
    # Coordinates in the original fitted image; y increases downward.
    bounds: tuple = FIT_VIEW

    def _set(self, x, y, span):
        x = min(max(x, 0.0), 1.0 - span)
        y = min(max(y, 0.0), 1.0 - span)
        bounds = (x, y, x + span, y + span)
        changed = bounds != self.bounds
        self.bounds = bounds
        return changed

    def zoom(self, wheel, anchor):
        x0, y0, x1, _ = self.bounds
        old = x1 - x0
        # Bound the exponent as well as the final scale for unusual input devices.
        span = min(1.0, max(1.0 / MAX_ZOOM, old / (1.25 ** max(-32, min(32, wheel)))))
        ax, ay = (min(1.0, max(0.0, float(v))) for v in anchor)
        return self._set(x0 + ax * (old - span), y0 + ay * (old - span), span)

    def pan(self, dx, dy):
        """Drag distances as fractions of the panel's width and height."""
        x0, y0, x1, _ = self.bounds
        span = x1 - x0
        return self._set(x0 - dx * span, y0 - dy * span, span)

    def fit(self):
        changed = self.bounds != FIT_VIEW
        self.bounds = FIT_VIEW
        return changed

    def from_plot(self, bounds):
        """Accept fractional native input, retaining a square normalized crop."""
        if not all(isfinite(v) for v in bounds):
            return False
        x0, y0, x1, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            return False
        span = min(1.0, max(1.0 / MAX_ZOOM, min(x1 - x0, y1 - y0)))
        # Ignore sub-ulp differences from the y-axis sign conversion.
        if all(abs(a - b) < 1e-12 for a, b in zip(bounds, self.bounds)):
            return False
        return self._set((x0 + x1 - span) / 2, (y0 + y1 - span) / 2, span)

    def preview(self, rendered, width, height):
        """Visible image rectangle and UVs, clipped rather than edge-stretched."""
        x0, y0, x1, y1 = self.bounds
        rx0, ry0, rx1, ry1 = rendered
        left, top = max(x0, rx0), max(y0, ry0)
        right, bottom = min(x1, rx1), min(y1, ry1)
        if right <= left or bottom <= top:
            return None
        return dict(
            pmin=((left - x0) / (x1 - x0) * width,
                  (top - y0) / (y1 - y0) * height),
            pmax=((right - x0) / (x1 - x0) * width,
                  (bottom - y0) / (y1 - y0) * height),
            uv_min=((left - rx0) / (rx1 - rx0), (top - ry0) / (ry1 - ry0)),
            uv_max=((right - rx0) / (rx1 - rx0), (bottom - ry0) / (ry1 - ry0)),
        )

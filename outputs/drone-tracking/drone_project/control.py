"""Simplified image-based proportional controller and persistent target lock."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Detection:
    track_id: int
    class_id: int
    confidence: float
    xyxy: tuple


class TargetLock:
    def __init__(self, class_ids=(0, 1), target_id=None):
        self.classes = set(class_ids)
        self.target_id = target_id
        self.acquired = target_id is not None

    def select(self, detections):
        eligible = [d for d in detections if d.class_id in self.classes]
        if not self.acquired and eligible:
            chosen = max(eligible, key=lambda d: (d.xyxy[2]-d.xyxy[0])*(d.xyxy[3]-d.xyxy[1]))
            self.target_id, self.acquired = chosen.track_id, True
        return next((d for d in eligible if d.track_id == self.target_id), None)


class IBVSController:
    """Normalized screen error -> bounded yaw/vertical velocity in a toy camera plant.

    This is a proportional visual-servo approximation, not a calibrated 6-DOF IBVS
    interaction-matrix controller. Positive yaw pans the camera right; positive vz
    moves its optical center up. Forward motion stays zero (no depth estimate).
    """
    def __init__(self, gain=1.0, max_yaw=.6, max_vertical=.4, deadband=.025):
        if not all(math.isfinite(v) for v in (gain, max_yaw, max_vertical, deadband)):
            raise ValueError('Controller parameters must be finite.')
        if gain <= 0 or max_yaw <= 0 or max_vertical <= 0 or not 0 <= deadband < 1:
            raise ValueError('Invalid controller gains or limits.')
        self.gain, self.max_yaw, self.max_vertical, self.deadband = gain, max_yaw, max_vertical, deadband

    def command(self, box, width, height):
        if width <= 0 or height <= 0:
            raise ValueError('Frame dimensions must be positive.')
        if box is None:
            return dict(state='lost', ex=0., ey=0., yaw=0., vertical=0., forward=0.)
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError('Invalid bounding box.')
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Bounding box has no area.')
        ex, ey = ((x1+x2)/2-width/2)/(width/2), ((y1+y2)/2-height/2)/(height/2)
        clamp = lambda v, limit: max(-limit, min(limit, v))
        yaw = 0. if abs(ex) < self.deadband else clamp(self.gain*ex, self.max_yaw)
        vertical = 0. if abs(ey) < self.deadband else clamp(-self.gain*ey, self.max_vertical)
        return dict(state='tracking', ex=ex, ey=ey, yaw=yaw, vertical=vertical, forward=0.)


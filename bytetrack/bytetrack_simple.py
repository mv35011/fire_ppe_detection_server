"""
Simplified ByteTrack implementation for person tracking
No external dependencies required
"""

import numpy as np
from collections import defaultdict


class SimpleTrack:
    """Simple track object"""
    track_id_counter = 0

    def __init__(self, bbox, score):
        self.track_id = SimpleTrack.track_id_counter
        SimpleTrack.track_id_counter += 1
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.is_activated = True

    def update(self, bbox, score):
        """Update track with new detection"""
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        """Mark track as missed in current frame"""
        self.time_since_update += 1
        if self.time_since_update > 30:  # Remove after 30 frames
            self.is_activated = False

    @property
    def tlbr(self):
        """Get bbox in [x1, y1, x2, y2] format"""
        return np.array(self.bbox)


class SimpleBYTETracker:
    """Simplified BYTE tracker implementation"""

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracks = []
        self.lost_tracks = []
        self.frame_id = 0

    def update(self, detections, img_shape):
        """Update tracker with new detections

        Args:
            detections: numpy array of shape (N, 5) with [x1, y1, x2, y2, score]
            img_shape: tuple of (height, width)

        Returns:
            list of active tracks
        """
        self.frame_id += 1

        if len(detections) == 0:
            # Mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.is_activated]
            return []

        # Split detections by score
        high_detections = []
        low_detections = []

        for det in detections:
            if det[4] >= self.track_thresh:
                high_detections.append(det)
            else:
                low_detections.append(det)

        # Match high score detections with existing tracks
        matched_tracks = []
        unmatched_dets = list(range(len(high_detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(self.tracks) > 0 and len(high_detections) > 0:
            # Calculate IoU matrix
            iou_matrix = self._calculate_iou_matrix(
                [t.bbox for t in self.tracks],
                [d[:4] for d in high_detections]
            )

            # Match using Hungarian algorithm (simplified greedy matching)
            for track_idx in range(len(self.tracks)):
                if track_idx not in unmatched_tracks:
                    continue

                best_match = -1
                best_iou = self.match_thresh

                for det_idx in unmatched_dets:
                    if iou_matrix[track_idx, det_idx] > best_iou:
                        best_iou = iou_matrix[track_idx, det_idx]
                        best_match = det_idx

                if best_match >= 0:
                    # Update matched track
                    self.tracks[track_idx].update(
                        high_detections[best_match][:4],
                        high_detections[best_match][4]
                    )
                    matched_tracks.append(track_idx)
                    unmatched_dets.remove(best_match)
                    unmatched_tracks.remove(track_idx)

        # Create new tracks for unmatched high score detections
        for det_idx in unmatched_dets:
            det = high_detections[det_idx]
            new_track = SimpleTrack(det[:4], det[4])
            self.tracks.append(new_track)

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.is_activated]

        return self.tracks

    def _calculate_iou_matrix(self, bboxes1, bboxes2):
        """Calculate IoU matrix between two sets of bboxes"""
        n1 = len(bboxes1)
        n2 = len(bboxes2)
        iou_matrix = np.zeros((n1, n2))

        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                iou_matrix[i, j] = self._calculate_iou(bbox1, bbox2)

        return iou_matrix

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        # Get intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Get areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate IoU
        union = area1 + area2 - intersection
        if union <= 0:
            return 0.0

        return intersection / union
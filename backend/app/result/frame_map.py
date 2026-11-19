"""What the result viewer draws: the two skeletons in playback order.

The scorer's frame pairs (Result.debug['path']) are made for comparing poses: one user frame can go with many
coach frames and the other way round. Drawn as they are, the coach skips frames (17 -> 30 in one step) or
stands still, the more so the more the two movements differ; a movement only the coach makes is never shown,
and a difference in speed cannot be seen because the two always move together. They are not sent.

The response carries both skeletons in the scorer's playback order instead (motion_scoring step6_playback:
per segment both at their own real speed and in step again at every checkpoint; where only one of them
moves, the other waits). The two sequences are equally long and frame k of the student belongs with frame k
of the coach:
  frameMap    per frame k: coachFrame = the coach frame to draw (k itself), and kind = who moves there:
              together | coachWaits (only the user moves) | coachAhead (only the coach moves)
  ranges      a segment's studentRange is its time on the playback timeline (a missing segment has a length
              there: the coach is shown making the movement); coachRange stays the time on the coach's own
              video. The times on the user's own video are not sent: they are the scorer's student_t, which
              the llm stage reads.
"""


def in_playback_order(pb, student, coach, segments):
    """Put the response in the order of the playback pb (motion_scoring Result.playback()), in place: the frames
    of the two skeleton sequences and the segments' studentRange. -> the frameMap of that order."""
    student['frames'] = [student['frames'][i] for i in pb['student']]
    coach['frames'] = [coach['frames'][i] for i in pb['coach']]
    coach['fps'] = student['fps']                          # the playback clock is the student's frame rate
    for seg, shown in zip(segments, playback_ranges(pb, student['fps'])):
        if shown:
            seg['studentRange'] = shown
        else:
            seg.pop('studentRange', None)
    return frame_map(pb)


def frame_map(pb):
    """frameMap of a playback. The kind names are the ones the response has always used."""
    c, s = pb['coach'], pb['student']
    kind = ['together'] * len(c)
    for k in range(1, len(c)):
        coach_moves, user_moves = c[k] != c[k - 1], s[k] != s[k - 1]
        if coach_moves != user_moves:
            kind[k] = 'coachAhead' if coach_moves else 'coachWaits'
    return dict(coachFrame=[float(k) for k in range(len(c))], kind=kind)


def playback_ranges(pb, fps):
    """[start, end] (s) of every segment on the playback timeline; None for a segment with nothing left to play.

    The frontend shows the segment with start <= frame / fps < end (utils/segments.ts segmentAtTime) and none
    otherwise. So a segment lasts from its first frame until the next segment starts -- the few frames between
    two of them (the turn at a reversal) stay with the one before -- and the times are frame / fps as the
    frontend computes them, not rounded: from the first segment on every frame has its segment, and a seek to
    `start` lands on the segment's first frame."""
    out, nxt = [], None
    for span in reversed(pb['segments']):
        out.append(span and [span[0] / fps, (span[1] + 1 if nxt is None else nxt) / fps])
        nxt = span[0] if span else nxt
    return out[::-1]

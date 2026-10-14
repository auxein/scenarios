"""Fitness (design doc §5): the measurements each episode reports and the Auxein `Aggregator` that turns them into one shaped score.

The score is maximised: mainly the box's progress toward the target, a delivery bonus and a time bonus, and a small approach term.
The raw measurements (delivered, steps, final distance, approach) are recorded per episode so they can be charted separately and
the run re-judged with other weights (`RunReader.reaggregate`) without re-simulating.
"""

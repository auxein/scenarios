"""Offline rendering helpers: matplotlib figures to MP4 through ffmpeg.

ffmpeg comes from the `imageio-ffmpeg` wheel (no system install), and matplotlib's `FFMpegWriter` is pointed at it, so `uv sync`
is enough to render on a Mac and in CI. Shared pieces: the writer, frame timing, titles and overlays (generation number, a
fitness curve with a moving marker), and composing clips into a montage (design doc §7).
"""

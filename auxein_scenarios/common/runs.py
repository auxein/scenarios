"""Run directories: where a training run, its Auxein recording and the videos rendered from it live.

A run directory holds Auxein's recording (`metadata.json`, `events.sqlite`, checkpoints), the config it was trained from and the
renders made from it afterwards. Helpers to name, create and find them, and to open the recording with `auxein.open_run`.
"""

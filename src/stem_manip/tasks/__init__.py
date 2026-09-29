"""Task registrations for the project (loaded through the `isaaclab.tasks` entry point)."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])

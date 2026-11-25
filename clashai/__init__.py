"""A Clash Royale agent trained to play one deck, Piggies EQ, on the crforge simulator."""

from .paths import apply_cpu_cap

# At most 16 logical processors (8 cores) unless CLASHAI_MAX_THREADS says otherwise: see
# clashai.paths.apply_cpu_cap. First, so the numeric libraries' pools are sized before they load.
apply_cpu_cap()

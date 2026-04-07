"""TIE/DLOC: Network-to-Network service simulator for cascading failure analysis.

Usage:
    import tiedloc

    # From JSON config
    config = tiedloc.SimulationConfig.from_json("examples/small_ba.json")
    results = tiedloc.run(config)

    # Programmatic
    config = tiedloc.SimulationConfig(
        topology="Barabasi Albert Scale-Free Network",
        topology_params={"num_of_nodes": 50, "new_node_to_existing_nodes": 3},
    )
    results = tiedloc.run(config)
    print(results.stats["Recoverability"])
"""

__version__ = "2.0.0"

from tiedloc.api import SimulationConfig, SimulationResults, run, sweep

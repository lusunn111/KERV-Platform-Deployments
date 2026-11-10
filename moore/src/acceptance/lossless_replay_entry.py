"""Fixed-input replay with the explicitly registered lossless storage backend."""
from experiments.robot.libero.run_lossless_libero import register
register()
from acceptance.replay_entry import main
main()

# Colab V5E1 connection

Date: 2026-10-09.
Status: connection passed; scientific execution did not occur.

Two earlier V6E1 requests remained pending until the reviewer closed them.
The cause remains unknown.
The reviewer then selected V5E1 in a new empty Colab notebook.
The browser showed a connected TPU after approximately 150.211 seconds.
The official CLI reported one V5E1 assignment with the Standard shape and TPU variant.

The reviewer executed no notebook cell and installed no package.
The reviewer terminated this runtime through the same notebook.
The official CLI then reported no server session, zero active assignments, and zero usage per hour.
These observations completed within the original 600-second limit.
All five local CLI process groups closed.
The empty Drive notebook remains as a connection record.

This result shows that V5E1 connected during this observation.
It does not establish the cause of the V6E1 waits or future device availability.
It supplies no numerical, compiler, memory, or performance result.
Accepted milestones remain three of eight.

The existing scientific owner admits only V6E1 browser records.
An explicit V5E1 extension requires separate source and lifecycle tests before scientific execution.
Each later result must retain its actual hardware type.

Refer to the [selected result](../../experiments/2026-10-04-bitsandbytes-tpu/results/colab-v5-connection.json).
Raw provider logs remain outside Git.

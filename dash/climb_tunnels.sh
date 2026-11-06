#!/bin/bash
# Local port for the hill-climb coordinator: 16200 -> arena9:6200 (reconnects on drop). nicky is no longer used.
while true; do ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 16200:localhost:6200 arena9-yarn; sleep 5; done

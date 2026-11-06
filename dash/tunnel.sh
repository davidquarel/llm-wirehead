#!/bin/bash
# Reverse tunnel: ap (cute.sus.cat) 127.0.0.1:8771 -> this machine's dashboard on 127.0.0.1:8771. Reconnects on drop.
while true; do
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -R 127.0.0.1:8771:127.0.0.1:8771 ap
  sleep 5
done

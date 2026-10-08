#!/bin/bash
# track4_sdmonly_spark_failover_pull.sh RUN HOST PORT - runs ON THE SPARK. Copies a rented box's newest checkpoint of
# RUN to ~/sdmonly_base/RUN/ so the base run can resume on the Spark if the box dies (the Spark is the failover).
#
# <claudes_code_comments>
# ** Function List **
# say(msg) - timestamped line to stdout (the caller redirects to a log)
# boxrun(cmd) - one ssh command on the box with the Spark's pull key
# one_round() - snapshot, pull, verify and keep the newest two copies
#
# ** Technical Review **
# - The trainer rewrites ck/RUN/last.pt every --ckpt-every steps, and a copy over a link of about 0.9 MB/s takes longer
#   than that, so the copy never reads last.pt directly. Each round first makes a box-side snapshot (cp last.pt to
#   snap.pt, then sha256 of snap.pt), and only the snapshot travels. A new snapshot is made only when last.pt is newer
#   than the previous one and no pull is in progress, so one copy runs at a time per box.
# - The pull is rsync --partial (a dropped link resumes from the bytes already on the Spark). After it the sha256 of the
#   Spark copy must equal the box-side sha256; only then is it renamed to step_<N>.pt and logged VERIFIED. The two
#   newest verified copies are kept; older ones are deleted by explicit file name.
# - The step is read from the run's log on the box (the last "step" event), which is close to but not exactly the
#   checkpoint's step; the checkpoint itself records its step, which a resume reads.
# - Loops every 10 minutes until killed. Uses only the Spark's disk and network, never its GPU.
# </claudes_code_comments>
set -u
RUN=$1; HOST=$2; PORT=$3
KEY=$HOME/.ssh/headclimb_pull
D=$HOME/sdmonly_base/$RUN; mkdir -p "$D"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT"
say() { echo "$(date -u +%FT%TZ) $*"; }
# Vast's sshd refuses a login now and then ("if authentication fails, try again after a few seconds"), so an ssh
# failure (exit 255) is retried up to 4 times, 20 s apart; any other exit code is the remote command's own answer.
boxrun() {
  local rc out try
  for try in 1 2 3 4; do
    out=$($SSH root@$HOST "$1" 2>>"$D/ssh_err.log"); rc=$?
    [ $rc = 255 ] || break
    say "ssh failed (exit 255), try $try of 4" >&2; sleep 20
  done
  [ $rc = 0 ] || say "box command exit $rc (ssh errors in $D/ssh_err.log)" >&2
  printf '%s' "$out"; return $rc
}

one_round() {
  # 1. snapshot on the box when last.pt is newer than the last snapshot, but ONLY when no pull is part-way here:
  #    a checkpoint lands about every 70 min and a copy can take longer, so refreshing mid-pull would never finish
  local info FRESH=1
  [ -f "$D/incoming.pt" ] && FRESH=0
  info=$(boxrun "cd /root/settle/ck/$RUN 2>/dev/null || exit 3
    [ -f last.pt ] || exit 4
    if [ ! -f snap.pt ] || { [ $FRESH = 1 ] && [ last.pt -nt snap.pt ]; }; then
      cp last.pt snap.tmp && mv snap.tmp snap.pt && sha256sum snap.pt | cut -c1-64 > snap.sha
      grep -o '\"event\": \"step\", \"step\": [0-9]*' /root/settle/runs/$RUN.log | tail -1 | grep -o '[0-9]*\$' > snap.step
    fi
    echo \$(cat snap.sha) \$(cat snap.step) \$(stat -c %s snap.pt)")
  [ -n "$info" ] || { say "no checkpoint on the box yet"; return; }
  set -- $info; local SHA=$1 STEP=$2 SIZE=$3
  [ -f "$D/step_$STEP.pt" ] && { say "step $STEP already verified here"; return; }
  # 2. pull the snapshot (resumable)
  say "pulling step $STEP ($SIZE bytes)"
  rsync -a --partial --inplace -e "$SSH" root@$HOST:/root/settle/ck/$RUN/snap.pt "$D/incoming.pt" || { say "pull interrupted, resumes next round"; return; }
  # 3. verify, then keep it
  local GOT; GOT=$(sha256sum "$D/incoming.pt" | cut -c1-64)
  if [ "$GOT" = "$SHA" ]; then
    mv "$D/incoming.pt" "$D/step_$STEP.pt"; echo "$SHA" > "$D/step_$STEP.sha"
    say "VERIFIED step $STEP sha $SHA"
    ls -1t "$D"/step_*.pt | tail -n +3 | while read -r old; do rm -f "$old" "${old%.pt}.sha"; say "dropped $(basename "$old")"; done
  else
    say "SHA MISMATCH step $STEP (box $SHA, here $GOT); the snapshot may have changed, retrying next round"
    rm -f "$D/incoming.pt"
  fi
}

while true; do one_round; sleep 600; done

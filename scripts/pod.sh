#!/usr/bin/env bash
# scripts/pod.sh — one tool to bring up / reach / stop our RunPod GPU pods.
#
# We keep a few IDENTICAL pods (named pitch3d-pro4500*) attached to the SAME
# persistent network volume (EU-RO-1; /workspace survives stop/start) but living
# on DIFFERENT host machines. GPU starvation ("not enough free GPUs on the host")
# is HOST-specific, so when one pod refuses to resume the fix is simply to try the
# next identical pod — which is exactly what `up` does, automatically.
#
# Subcommands:
#   status | (none)   list every pod (id / state / name / ssh endpoint) + spend
#   up                ensure ONE pod is RUNNING — reuse a live one, else resume each
#                     EXITED pod in turn until a host places the GPU; prints an ssh cmd
#   ssh [cmd...]      ssh into the running pod (run cmd if given); does NOT start one
#   down              stop ALL running pods — call this whenever the box is idle ($)
#
# Auth: runpodctl reads ~/.runpod/config.toml (`apikey`). If empty, the key also
# lives in the RunPod MCP server env inside ~/.claude.json — see
# docs/runpod-runbook.md §0c for the one-liner that repopulates the config.
#
# Env overrides:
#   RUNPODCTL       runpodctl path (default: PATH, else ~/.local/bin/runpodctl)
#   POD_SSH_KEY     ssh private key (default: ~/.ssh/id_ed25519_runpod)
#   POD_NAME_GLOB   jq regex picking our pods (default: "pitch3d")
#   START_TIMEOUT   per-pod seconds to wait for an endpoint after resume (default 90)
set -euo pipefail

RUNPODCTL="${RUNPODCTL:-$(command -v runpodctl || echo "$HOME/.local/bin/runpodctl")}"
POD_SSH_KEY="${POD_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
POD_NAME_GLOB="${POD_NAME_GLOB:-pitch3d}"
START_TIMEOUT="${START_TIMEOUT:-90}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
die(){ echo "pod.sh: $*" >&2; exit 1; }
[ -x "$RUNPODCTL" ] || die "runpodctl not found (set RUNPODCTL=/path/to/runpodctl)"
command -v jq >/dev/null || die "jq is required"
rp(){ "$RUNPODCTL" "$@"; }

# All of our pods as a compact JSON array, filtered by name.
pods_json(){
  rp pod list -a -o json 2>/dev/null \
    | jq -c --arg g "$POD_NAME_GLOB" '[.[] | select((.name//"")|test($g))]'
}

# One pod's endpoint -> sets globals EP_IP / EP_PORT / EP_ST (empty if not up).
pod_ep(){
  EP_IP=""; EP_PORT=""; EP_ST=""
  local j; j="$(rp pod get "$1" -o json 2>/dev/null)" || return 0
  [ -n "$j" ] || return 0
  EP_IP="$(jq -r '.ssh.ip // ""'              <<<"$j")"
  EP_PORT="$(jq -r '.ssh.port // "" | tostring' <<<"$j")"
  EP_ST="$(jq -r '.desiredStatus // ""'       <<<"$j")"
}

# Poll one pod up to START_TIMEOUT for a usable RUNNING endpoint.
wait_endpoint(){
  local id="$1" waited=0
  while [ "$waited" -lt "$START_TIMEOUT" ]; do
    pod_ep "$id"
    [ "$EP_ST" = "RUNNING" ] && [ -n "$EP_IP" ] && [ -n "$EP_PORT" ] && return 0
    sleep 6; waited=$((waited+6))
  done
  return 1
}

print_ssh(){  # $1=ip $2=port
  echo
  echo "  ssh -i $POD_SSH_KEY -p $2 ${SSH_OPTS[*]} root@$1"
  echo
  echo "  (or: scripts/pod.sh ssh '<cmd>')   — stop when idle: scripts/pod.sh down"
}

ok_ssh(){ pod_ep "$1"; echo "[up] READY: $1 @ $EP_IP:$EP_PORT"; print_ssh "$EP_IP" "$EP_PORT"; }

fail_migrate(){
  cat >&2 <<'EOF'
[up] FAILED — none of our pods could place a GPU on their host right now.
     Every host is starved. Options, in order:
       * wait a few minutes and retry:  scripts/pod.sh up
       * ask RunPod support for a GPU migration of one pod to a host with free
         RTX PRO 4500 Blackwell stock (console support chat), then retry.
       * create a fresh pod (Blackwell needs the *deprecated* `runpodctl create
         pod` + a cu128 image) — see docs/runpod-runbook.md §2.
EOF
}

cmd_up(){
  local list id
  list="$(pods_json)" || die "cannot list pods (auth? run: $RUNPODCTL me)"
  [ "$(jq 'length' <<<"$list")" -gt 0 ] || die "no pods match /$POD_NAME_GLOB/ (set POD_NAME_GLOB)"

  # 1) A pod already desired-RUNNING: wait for its endpoint — never start a 2nd.
  for id in $(jq -r '.[]|select(.desiredStatus=="RUNNING")|.id' <<<"$list"); do
    echo "[up] $id already desired-RUNNING; waiting for endpoint ..."
    if wait_endpoint "$id"; then ok_ssh "$id"; return 0; fi
    echo "[up]   $id RUNNING but no endpoint after ${START_TIMEOUT}s; trying others"
  done

  # 2) Resume each EXITED pod until a host places the GPU.
  for id in $(jq -r '.[]|select(.desiredStatus!="RUNNING")|.id' <<<"$list"); do
    echo "[up] resuming $id ..."
    if ! rp pod start "$id" >"$TMP" 2>&1; then
      echo "[up]   refused: $(tr '\n' ' ' <"$TMP" | tail -c 200)"; continue
    fi
    if wait_endpoint "$id"; then ok_ssh "$id"; return 0; fi
    echo "[up]   $id got no endpoint in ${START_TIMEOUT}s; stopping it (avoid idle billing) & trying next"
    rp pod stop "$id" >/dev/null 2>&1 || true
  done

  fail_migrate; return 1
}

cmd_ssh(){
  local list id found=""
  list="$(pods_json)" || die "cannot list pods"
  for id in $(jq -r '.[]|select(.desiredStatus=="RUNNING")|.id' <<<"$list"); do
    pod_ep "$id"
    [ -n "$EP_IP" ] && [ -n "$EP_PORT" ] && { found="$id"; break; }
  done
  [ -n "$found" ] || die "no RUNNING pod with an endpoint — run: scripts/pod.sh up"
  echo "[ssh] $found @ $EP_IP:$EP_PORT" >&2
  exec ssh -i "$POD_SSH_KEY" -p "$EP_PORT" "${SSH_OPTS[@]}" "root@$EP_IP" "$@"
}

cmd_down(){
  local list id any=""
  list="$(pods_json)" || die "cannot list pods"
  for id in $(jq -r '.[]|select(.desiredStatus=="RUNNING")|.id' <<<"$list"); do
    printf '[down] stopping %s ... ' "$id"
    if rp pod stop "$id" >/dev/null 2>&1; then echo "ok"; else echo "FAILED"; fi
    any=1
  done
  [ -n "$any" ] || echo "[down] no RUNNING pods — nothing to stop"
  echo "[down] state now:"; cmd_status
}

cmd_status(){
  local me list ep
  me="$(rp me -o json 2>/dev/null || true)"
  [ -n "$me" ] && echo "account $(jq -r '.email' <<<"$me")  balance \$$(jq -r '(.clientBalance*100|floor)/100' <<<"$me")  spend \$$(jq -r '.currentSpendPerHr' <<<"$me")/hr"
  list="$(pods_json)" || die "cannot list pods"
  jq -r '.[] | "\(.id)\t\(.desiredStatus)\t\(.name)"' <<<"$list" | while IFS=$'\t' read -r id st name; do
    pod_ep "$id"
    if [ -n "$EP_IP" ] && [ -n "$EP_PORT" ]; then ep="$EP_IP:$EP_PORT"; else ep="-"; fi
    printf '  %-16s %-8s %-30s %s\n' "$id" "$st" "$name" "$ep"
  done
}

case "${1:-status}" in
  up)            shift; cmd_up "$@";;
  ssh)           shift; cmd_ssh "$@";;
  down)          shift; cmd_down "$@";;
  status)        cmd_status;;
  -h|--help|help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//';;
  *)             die "unknown subcommand: $1 (try: status | up | ssh | down)";;
esac

#!/usr/bin/env bash
# Spike 4: can workers bypass the router? Run against a real cluster (not tested in the build sandbox).
# Usage: ./run_test.sh            (uses the current kubectl context; creates and deletes spike4 namespaces)
set -u
cd "$(dirname "$0")"
K=${KUBECTL:-kubectl}
probe() {  # probe <label> <url> : prints "<label>: REACHED <body>" or "<label>: BLOCKED"
  out=$($K -n agents exec worker -- python -c "
import urllib.request,sys
try: print(urllib.request.urlopen('$2', timeout=4).read().decode()[:80])
except Exception as e: print('ERR', type(e).__name__); sys.exit(1)" 2>&1) && echo "$1: REACHED  $out" || echo "$1: BLOCKED  ($out)"
}
cleanup() { $K delete ns agents router external --ignore-not-found --wait=false >/dev/null 2>&1; }
trap cleanup EXIT
$K apply -f 00-base.yaml >/dev/null || exit 1
$K -n router rollout status deploy/router --timeout=180s >/dev/null
$K -n external rollout status deploy/provider --timeout=180s >/dev/null
$K -n agents wait --for=condition=Ready pod/worker --timeout=180s >/dev/null
PROV_CIP=$($K -n external get svc provider -o jsonpath='{.spec.clusterIP}')
PROV_POD=$($K -n external get pod -l app=provider -o jsonpath='{.items[0].status.podIP}')
ROUTER=http://router.router.svc.cluster.local:8080
targets() {
  probe "via router (allowed)        " "$ROUTER/via"
  probe "provider by service name    " "http://provider.external.svc.cluster.local:8080/"
  probe "provider by cluster IP      " "http://$PROV_CIP:8080/"
  probe "provider by pod IP          " "http://$PROV_POD:8080/"
  probe "public internet (1.1.1.1)   " "http://1.1.1.1/"
}
echo "== BASELINE: no policy. Expect every direct probe REACHED (bypass possible) =="
targets
$K apply -f 10-netpol.yaml >/dev/null
sleep 10   # allow the plugin to program rules
echo; echo "== WITH POLICY. Expect via-router REACHED, every direct probe BLOCKED =="
targets
echo; echo "== PASS CRITERIA =="
echo "1. Baseline: direct probes REACHED, proving the test can detect a bypass."
echo "2. With policy: via-router REACHED (and shows 'router-ok -> provider-reached-directly'), all direct probes BLOCKED."
echo "3. Record the network plugin: $($K get pods -n kube-system -o name | grep -i -E 'calico|cilium|kindnet|flannel|weave|antrea' | head -3 | tr '\n' ' ')"
echo "4. 'Logged': standard NetworkPolicy blocks silently. Check whether your plugin can log denials (see README) before calling the exit criterion met."

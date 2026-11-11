# Spike 4: egress enforcement (prepared; NOT run)

**Status: not run.** The build sandbox could not start a Kubernetes node: a kind v0.33 node (Kubernetes v1.34, Docker 29.9.2, cgroup v1 host, kernel 6.18 on Firecracker) failed with `runc create failed: can't get final child's PID from pipe: EOF` twice, once with a 4 minute wait. So these manifests and the script are untested against any cluster. Run them in your cluster.

## Question
Can a worker (or an external harness) bypass the router to reach a model provider, and can we block that with network policy and see the attempt?

## Files
- `00-base.yaml`: namespaces `agents`, `router`, `external`; a stub router, a stub provider, a worker pod. Uses `python:3.13-slim` only.
- `10-netpol.yaml`: deny all egress from `agents` except the router on 8080 and cluster DNS; accept traffic into `external` only from `router`.
- `run_test.sh`: runs a baseline (no policy) and then the same probes with policy.

## Run
```bash
cd spikes/04-egress && ./run_test.sh
```

## Exit criteria (from the tools list)
A bypass attempt is blocked **and logged**; note which network plugin enforces it.

## Known gaps to check
- **Logging.** Standard `NetworkPolicy` does not log denied traffic. Whether denials can be logged depends on the plugin (for example flow logs in Cilium or Calico, or a node-level log rule). Decide the detection route, since the decision record says detection and alerting are still open.
- **Enforcement.** Some plugins accept the objects but do not enforce them. The baseline-versus-policy design exists to catch that.
- **DNS.** The policy allows DNS to kube-system, so DNS tunnelling is not blocked by this policy.
- **Other egress paths:** a node-local proxy, `hostNetwork` pods, and IPv6 are not tested. Add them once the basic result is in.
- **Customer networks:** repeat on a sample customer cluster; the egress rule must also be expressible in whatever their network team runs.

# Which Intel GPUs this catalog covers

Stage 5 of the router, in detail. The question this file answers is narrow: **is there a
skill here that knows about the part in front of me?** It is not a hardware guide.

## Covered

| Family | What covers it |
|---|---|
| Arc (consumer, discrete) | **xpu-discover** inventories it; the runtime skills serve on it |
| Arc Pro, including the B-series | as above, plus **model-config-recommend**, which carries a per-SKU table for the B-series, and **xpu-system-setup**, which owns the Battlemage prerequisites |
| Data Center GPU Max | **xpu-discover** inventories it |

The B-series parts are the ones this catalog has the most specific knowledge of: a config
recommender that knows their memory and compute tiers, a setup skill that knows what a fresh
host is missing, and runtime skills that name the Battlemage-specific environment settings.
Those settings belong to the runtime skills — the router does not repeat them.

## Not covered

Anything else, and in particular any Intel GPU announced after the skills in this catalog
were written — **Crescent Island** is the current example.

There is no data here for an uncovered part: no memory figure, no compute tier, no known
driver or runtime minimum, and no evidence that the images the runtime skills name support
it at all. The correct output is to say so.

**Do not extrapolate.** Reasoning from a B-series row to an unlisted part produces a fit
verdict, a quantization recommendation or a concurrency number that reads as authoritative
and rests on nothing. That failure is worse than no answer, because the user cannot see that
it was invented.

What to do instead:

1. Report that the catalog has no data for the part, and name what is missing.
2. Run **xpu-discover** anyway if the machine is in front of you — it reads what the driver
   reports rather than what a table says, so it is still truthful on an unlisted part.
3. If the driver enumerates the device and the user wants to proceed, say clearly that
   everything downstream is unverified on this hardware, and let them decide.

## Hardware that is not an Intel GPU

An NVIDIA or AMD target is stage 0, not stage 5: this catalog is Intel-only, and the right
answer is to say so without naming an Intel skill. A CUDA codebase that wants to move is a
different request again — **cuda-to-xpu-migration** to assess it, **xpu-port** to execute
the port.

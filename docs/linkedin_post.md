Permissive left turns produced 101 crossing conflicts per 15 minutes in my simulation. A flashing yellow arrow cut that in half and the intersection only dropped one LOS grade.

I built an open-source surrogate safety engine that reproduces the FHWA SSAM method (TTC, PET, DeltaV, conflict type) directly on PTV Vissim vehicle-record (.fzp) files, then used it to compare three left-turn phasing plans at a four-leg signalized intersection (10 random seeds each, 95% confidence intervals):

🔹 No-Build, permissive lefts: 119 conflicts per 15 min, LOS C (31 s/veh)
🔹 Build A, protected-only lefts: crossing conflicts -100%, but LOS E (57 s/veh) and rear-ends +57%
🔹 Build B, protected-permissive FYA: crossing conflicts -50%, severe conflicts (TTC ≤ 0.5 s) -82%, LOS D (42 s/veh)

Three things I learned building it:

1️⃣ Raw SSAM projection overcounts. It extends every vehicle in a straight line, so two left-turners mid-arc "point at" each other and get logged as a conflict. I added a path-verification step: keep an event only if the vehicles really shared space within max PET, or stayed on their projected course and braked. In the protected-only scenario, it removed 58% of raw TTC events.

2️⃣ Timing errors show up as safety data. My first run had a left-arrow yellow of 3.0 s. The trajectories showed a driver 44 m out at 32 mph with no comfortable stop entering on red. That is a dilemma zone, and the ITE formula says the yellow should be ~3.4 s. A conflict study is also a timing audit.

3️⃣ The choice of data structure has to fit the access pattern. An R*-tree sounds like the right spatial index, but rebuilt every 0.1 s frame it was 27x slower than a KD-tree at 5,000 vehicles. It does win for the static API index, where bounding-box queries return at p99 = 2.9 ms (cached, in-process).

The stack is Python 3.12, NumPy vectorised separating-axis TTC, scipy KD-tree / libspatialindex R*-tree, FastAPI + Redis, Docker, 55 pytest cases with analytically solved TTC/PET fixtures, and a Vissim COM batch runner for multi-seed runs.

One caveat: the trajectories come from a transparent IDM + NEMA dual-ring test bed I wrote, not a calibrated Vissim model. The numbers show the method, not a recommendation for a real site.

Repo: [link]

What filters do you apply to SSAM output before you trust a conflict count?

#TrafficEngineering #TransportationSafety #VISSIM #SSAM #TrafficSimulation #SignalTiming #Python #SystemDesign #Geospatial #HNTB

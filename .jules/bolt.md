# Bolt's Journal

## 2024-05-23 - Initial Setup
**Learning:** Performance optimization requires a structured approach.
**Action:** Follow the daily process: Profile, Select, Optimize, Verify, Present.

## 2024-05-23 - Lazy Screenshot Conversion
**Learning:** `mss` returns BGRA images. Converting to RGB immediately for every frame is wasteful if most frames are discarded. Deferring conversion until a change is detected saves significant CPU/RAM.
**Action:** Inspect data flow of high-frequency loops. Look for transformations that can be lazy-loaded or deferred.

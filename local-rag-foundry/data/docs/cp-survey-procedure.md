---
title: Cathodic Protection Survey
category: Maintenance
id: MAINT-013
---

# Cathodic Protection Survey

## Overview
Buried and submerged steel pipelines corrode electrochemically wherever bare
metal is in contact with soil or water. Cathodic protection (CP) counters this
by making the pipe the cathode of an electrical cell, either through sacrificial
anodes or an impressed-current system driven by a transformer-rectifier. A CP
survey verifies that this protection is actually working across the full route,
not just at the rectifier. Without regular surveys, a coating defect or a
failed anode can allow corrosion to progress for years, undetected, until it
threatens the wall thickness of the pipe.

## Key Safety Precautions
- CP test stations may carry induced AC voltage from nearby power lines or
  pipelines; always use insulated tools and gloves when opening a test post.
- Do not disconnect bonding cables on pipelines with induced current without
  authorisation — doing so can create a shock hazard at an unexpected location
  further along the line.
- Some test stations are located in traffic areas or near live electrical
  equipment; carry out a site-specific risk assessment before opening the box.

## How the System Works
A sacrificial-anode system relies on a more reactive metal (typically
magnesium or zinc) buried near the pipe and electrically bonded to it. The
anode corrodes preferentially, protecting the steel. An impressed-current
system instead uses a transformer-rectifier to force a small DC current from
buried anode beds into the soil and onto the pipe. Impressed-current systems
can protect much longer pipeline sections from a single rectifier but require
a continuous power supply and periodic checks that the rectifier itself has
not tripped or drifted out of its set output.

## Acceptance Criteria
- Pipe-to-soil potential more negative than -0.85 V, measured against a
  copper/copper-sulphate reference electrode. This is the standard "protected"
  threshold used across the industry.
- Avoid potentials more negative than -1.2 V. Over-protection is not merely
  wasteful; it can generate hydrogen at the pipe surface and cause coating
  disbondment, which paradoxically increases the risk of localised corrosion
  under the lifted coating.
- On lines with interference from other buried structures or foreign
  pipelines, additional criteria (such as a minimum 100 mV shift) may apply —
  check the pipeline-specific CP design report.

## Working Procedure
1. Connect the reference electrode to the soil directly above the pipeline,
   as close to the pipe centreline as access allows.
2. Measure the pipe-to-soil potential at each scheduled test post along the
   route.
3. Record both the "on" potential (with the CP system energised) and the
   "instant-off" potential (taken within a fraction of a second of
   interrupting the current, before the soil IR drop decays). The instant-off
   reading is the one compared against the -0.85 V criterion, since the "on"
   reading includes voltage drop through the soil that can mask
   under-protection.
4. Compare each reading against the protection criterion and flag any
   post that fails.
5. Investigate any post reading less negative than -0.85 V — this can
   indicate a coating holiday, a shielded area, stray current interference,
   or a failed bond.
6. Check the rectifier output current and voltage at the transformer-rectifier
   unit, and compare against the design output to confirm it has not drifted
   or tripped since the last survey.
7. Log all readings and trend them against previous surveys; a gradual
   decline in protection level over several surveys is often the first sign
   of anode depletion or a developing coating fault, well before any single
   reading fails outright.

## Common Issues
- A rectifier that has tripped on overload or lost mains power can go
  unnoticed for months if the survey interval is too long — this is why
  remote monitoring units are increasingly fitted at critical rectifiers.
- Construction activity near the pipeline (new buried services, fencing,
  cathodic bonds added by others) can create shielding or interference that
  shows up as an unexplained potential shift.
- Sacrificial anode beds eventually deplete; a consistent downward trend in
  protection level, even while still passing the criterion, should trigger a
  proactive anode replacement rather than waiting for an outright failure.

## Source Standard
Maintenance Standard MS-21, Section 2 — Cathodic Protection.

# What Savant's three per-swing numbers are

Discovery on the 2025 labeled whiff tails. Written by Task 4 of the plan.
Every number here is measured; the machine-readable version is
`discovery_2025.json` and the full ladder is its `ladder` key.

## The labeled set

Savant's per-swing tail endpoint returns, for each player-season, the ten worst
whiffs by each of four quantities (miss distance, x, y, z), under both the
pitcher and the batter view. Pooled over the 1,497 players in the 2025
leaderboards and deduplicated by `play_id`, that is 21,329 distinct swings.
7,587 of them were selected by exactly one list and 152 by all eight.

Joined to our own rows on (game_pk, batter, pitcher, miss distance rounded to
six places): **21,328 matched exactly, 1 unmatched, join rate 0.99995**. No row
needed the nearest-distance fallback. Twelve extra rows came out of the exact
merge because two of our swings shared a join key; those are deduplicated to one
row per `play_id` and counted rather than dropped silently.

All 21,328 are whiffs, which is what the endpoint's leaked SQL says they should
be. The one row that read as a foul before turned out to be a foul tip, which
Savant counts as a whiff: our whiff rate is 0.2346 with foul tips called contact
and 0.2565 with them called whiffs, against Savant's published 0.2569.

Free parameters: `y_com_ft = 0.0` and `tilt_sign = +1`. See the bottom of this
file for how each was chosen.

## y is the contact depth expressed as the ball's flight time. RECOVERED.

**r-squared 0.951, label agreement 0.914, slope 1.00, all-swing.**

The winning candidate is the cheapest one on the ladder:

    sav_y = 0.924 * (intercept depth / ball speed at that depth) - 17.92    (RHB)
    sav_y = 0.911 * (intercept depth / ball speed at that depth) - 18.34    (LHB)

with depth in inches, ball speed in inches per millisecond, and the result in
milliseconds. The raw depth in inches gives almost the same fit and its pooled
slope is 0.699 ms per inch, whose reciprocal is 1.43 inches per millisecond, or
81.3 mph. That is the ball's speed near the plate. So Savant's early/late number
is not a bat-relative geometry at all. It is how much earlier or later than an
ideal depth the bat met the ball, converted to time at the speed the ball was
travelling.

The coefficient on the time form is 0.92, not 1.00, and the intercept puts the
zero crossing at 19.4 ms of ball flight for right-handed batters and 20.1 ms for
left-handed ones, which at the tails' median ball speed of 1.40 inches per
millisecond (79.5 mph) is **27.2 and 28.2 inches of contact depth**. Savant's "on time" is contact about 27 inches in
front of the batter's own reference point, and the +/- 7 ms band is about +/- 10
inches of depth around it.

Two consequences worth recording. First, the sign convention falls out: deeper
contact is later, so positive is early, which is Savant's convention.
Second, Savant's own published histograms corroborate the definition. Over the
2025 batters, the y histogram's total equals the leaderboard's `n_swings` for
every single player, while the x and z histograms run about 0.15 percent high,
which is the exact rate at which our own intercept columns are null. Savant has
an x and a z on swings whose timing value does not exist, which is what a
definition built on the intercept depth predicts and what a definition built on
anything else does not.

Every other y candidate is a proxy for the same thing. `attack_direction`
reaches r-squared 0.902 on its own and `swing_length` correlates 0.89, because
on a whiff all three move together with how far out front the bat was.

## x is not recovered. MODELED.

**Best candidate r-squared 0.712, label agreement 0.864. Both gates missed.**

The ladder's best is `x_full_geometry`, a nine-feature per-stand linear model
over the intercept location, attack direction, bat speed, the ball's position at
the intercept depth and the strike-zone-relative height. The cheapest candidate
that comes close is `along_bat_plus_icpt` at 0.690 with four features.

What x is not: on the tails it runs from -163.5 to +143.9 inches while the miss
distance on those same swings never exceeds 57.5 inches, and only 16.9 percent
of rows satisfy |x| <= miss distance. A signed offset along a 34 inch bat at the
instant of closest approach cannot do that, and neither can any component of the
3D miss vector. Whatever x measures, it is not measured at the instant miss
distance is measured.

The single strongest public quantity is the lateral intercept distance
(r = 0.544 pooled, and the same sign for both stands because that column is
already body-relative), followed by miss distance itself at 0.452 and the
intercept depth at 0.375. The intercept depth mattering at all is the tell: a
purely lateral, purely at-contact quantity should not care how far out front the
bat was.

Task 4 Step 5's first expensive hypothesis, the bat-frame projection taken at
the instant the ball crosses the swing plane, **is not computable from the
published fields**. It needs the bat's position in three dimensions, and
Statcast publishes the intercept point's lateral offset and depth but no height,
so the plane cannot be anchored. Anchoring it at the ball's own position at the
intercept depth makes the crossing instant the intercept instant by
construction, which is degenerate. This is recorded as unreachable rather than
tried and failed.

## z is not recovered. MODELED.

**Best candidate r-squared 0.665, label agreement 0.708. Both gates missed.**

The ladder's best is `z_miss_decomp`, which is Task 4 Step 5's third hypothesis
(z as a component of the miss vector, `sign(z_above_plane) * min(|z_above_plane|,
miss_distance)`) plus the strike-zone-relative ball height, the pitch descent
angle and the miss distance. The hypothesis itself is refuted as a definition:
only 53.6 percent of tail rows satisfy |z| <= miss distance, so z is not a
component of the miss vector either. It earns its place as a feature, not as an
explanation.

z does behave like contact-instant geometry, which is what the spec expected.
The ball's height at the intercept depth correlates 0.724 with it and the
pitch's descent angle 0.700, both with the same sign for both stands: a higher,
flatter pitch means the swing passed under it, and Savant's positive z is
"under". The strike-zone-relative height correlates identically at 0.724.

What is missing is the bat's own height at contact, which Statcast does not
publish. Everything the ladder can reach is the ball's side of the difference,
so the fits saturate near 0.6.

The plan's own cheap z candidate, the ball's displacement out of the swing plane
between the intercept depth and the plate, is worthless: r = -0.128, r-squared
0.016. It is recorded here because the plan named it first.

## The two free parameters

**tilt_sign = +1 (the barrel below the hands).** Geometry did not settle it: see
`attack_direction_sign.txt` for three measurements that disagree. The data did.
Over a 9 by 2 grid of (batter reference depth, tilt sign), the x ladder's best
r-squared is 0.695 at +1 against 0.611 at -1, while z prefers -1 by 0.0030. The
x preference is 28 times larger and agrees with how MLB depicts swing path tilt.

**y_com_ft = 0.0 (the point of the plate).** x and y are exactly insensitive to
it, as the plan predicted. z varies from 0.5619 to 0.5506 across the whole grid,
monotonically, with its maximum at the grid edge, which means the criterion does
not identify the parameter rather than that -2.0 feet is right. z falls back
either way, so the choice cannot change a verdict. 0.0 is the interpretable
anchor and is what ships.

## What this means downstream

y applies to every swing directly, from fields present on all of them. x and z
are modeled: a LightGBM regressor fit on these tails with inverse-selection
weights, applied to whiffs only, and the Driveline outcome inversion for contact
swings. The tails are the ten worst per list per player-season, selected on the
label, so they are whiff-only evidence and are treated as whiff-only evidence.

One alternative was identified and deliberately not pursued, because the plan
bounds the search here and the verdict is final once written. For balls in play
the bat and the ball met, so z is near zero by construction; a model of the
bat's height at the intercept could be learned from in-play swings and
subtracted from the ball's height to give z on whiffs. That is a different
experiment from the one this plan specifies, and it is the first thing to try if
anyone reopens z.
